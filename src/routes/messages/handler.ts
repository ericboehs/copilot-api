import type { Context } from "hono"
import type { SSEStreamingApi } from "hono/streaming"

import consola from "consola"
import { streamSSE } from "hono/streaming"

import type { ChatCompletionsPayload } from "~/services/copilot/create-chat-completions"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import {
  createChatCompletions,
  type ChatCompletionChunk,
  type ChatCompletionResponse,
} from "~/services/copilot/create-chat-completions"

import {
  type AnthropicMessagesPayload,
  type AnthropicStreamState,
  type AnthropicStreamEventData,
} from "./anthropic-types"
import {
  translateToAnthropic,
  translateToOpenAI,
} from "./non-stream-translation"
import {
  translateChunkToAnthropicEvents,
  translateErrorToAnthropicErrorEvent,
} from "./stream-translation"

const MAX_STREAM_RETRIES = 3

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  const anthropicPayload = await c.req.json<AnthropicMessagesPayload>()
  consola.debug("Anthropic request payload:", JSON.stringify(anthropicPayload))

  const openAIPayload = translateToOpenAI(anthropicPayload)
  consola.debug(
    "Translated OpenAI request payload:",
    JSON.stringify(openAIPayload),
  )

  if (state.manualApprove) {
    await awaitApproval()
  }

  if (!openAIPayload.stream) {
    return handleNonStreamingRequest(c, openAIPayload)
  }

  consola.debug("Streaming response from Copilot")
  const thinkingRequested = anthropicPayload.thinking?.type === "enabled"
  return streamSSE(c, async (stream) => {
    await handleStreamingWithRetry(stream, openAIPayload, thinkingRequested)
  })
}

async function handleNonStreamingRequest(
  c: Context,
  openAIPayload: ChatCompletionsPayload,
) {
  const response = await createChatCompletions(openAIPayload)
  if (!isNonStreaming(response)) return c.json({})

  consola.debug("Non-streaming response:", JSON.stringify(response).slice(-400))
  const anthropicResponse = translateToAnthropic(response)
  consola.debug(
    "Translated Anthropic response:",
    JSON.stringify(anthropicResponse),
  )
  return c.json(anthropicResponse)
}

async function handleStreamingWithRetry(
  stream: SSEStreamingApi,
  openAIPayload: ChatCompletionsPayload,
  thinkingRequested: boolean,
) {
  let messageStartSentToClient = false
  let hasContentData = false

  for (let attempt = 1; attempt <= MAX_STREAM_RETRIES; attempt++) {
    try {
      const result = await consumeUpstreamStream(stream, openAIPayload, {
        skipMessageStart: messageStartSentToClient,
        thinkingRequested,
      })
      messageStartSentToClient = result.messageStartSent
      hasContentData = result.hasContentData
      return // Stream completed successfully
    } catch (err) {
      consola.error(
        `Upstream stream error (attempt ${attempt}/${MAX_STREAM_RETRIES}):`,
        err,
      )

      if (hasContentData || attempt >= MAX_STREAM_RETRIES) {
        const errorEvent = translateErrorToAnthropicErrorEvent(
          `Upstream connection lost (attempt ${attempt}/${MAX_STREAM_RETRIES}). The model may have timed out during processing.`,
        )
        await writeEvent(stream, errorEvent)
        return
      }

      consola.info(
        `No content sent yet, retrying (attempt ${attempt + 1}/${MAX_STREAM_RETRIES})...`,
      )
    }
  }
}

async function consumeUpstreamStream(
  stream: SSEStreamingApi,
  openAIPayload: ChatCompletionsPayload,
  options: { skipMessageStart: boolean; thinkingRequested: boolean },
) {
  const { skipMessageStart, thinkingRequested } = options
  let messageStartSent = skipMessageStart
  let hasContentData = false

  const response = await createChatCompletions(openAIPayload)

  if (isNonStreaming(response)) {
    consola.warn("Received non-streaming response for stream request")
    const anthropicResponse = translateToAnthropic(response)
    await writeEvent(stream, {
      type: "message_start",
      message: anthropicResponse,
    } as AnthropicStreamEventData)
    await writeEvent(stream, {
      type: "message_stop",
    })
    return { messageStartSent: true, hasContentData: true }
  }

  const streamState: AnthropicStreamState = {
    messageStartSent: false,
    contentBlockIndex: 0,
    contentBlockOpen: false,
    thinkingBlockOpen: false,
    thinkingRequested,
    toolCalls: {},
  }

  for await (const rawEvent of response) {
    consola.debug("Copilot raw stream event:", JSON.stringify(rawEvent))
    if (rawEvent.data === "[DONE]") break
    if (!rawEvent.data) continue

    const chunk = JSON.parse(rawEvent.data) as ChatCompletionChunk
    const events = translateChunkToAnthropicEvents(chunk, streamState)

    for (const event of events) {
      if (event.type === "message_start" && messageStartSent) {
        consola.debug("Skipping duplicate message_start on retry")
        continue
      }

      consola.debug("Translated Anthropic event:", JSON.stringify(event))
      await writeEvent(stream, event)

      if (event.type === "message_start") messageStartSent = true
      if (event.type === "content_block_delta") hasContentData = true
    }
  }

  return { messageStartSent, hasContentData }
}

async function writeEvent(
  stream: SSEStreamingApi,
  event: AnthropicStreamEventData,
) {
  await stream.writeSSE({ event: event.type, data: JSON.stringify(event) })
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")
