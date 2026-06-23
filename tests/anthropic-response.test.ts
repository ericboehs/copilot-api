import { describe, test, expect } from "bun:test"
import { z } from "zod"

import type {
  ChatCompletionChunk,
  ChatCompletionResponse,
} from "~/services/copilot/create-chat-completions"

import { type AnthropicStreamState } from "~/routes/messages/anthropic-types"
import { translateToAnthropic } from "~/routes/messages/non-stream-translation"
import { translateChunkToAnthropicEvents } from "~/routes/messages/stream-translation"

const anthropicUsageSchema = z.object({
  input_tokens: z.number().int(),
  output_tokens: z.number().int(),
})

const anthropicContentBlockTextSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
})

const anthropicContentBlockToolUseSchema = z.object({
  type: z.literal("tool_use"),
  id: z.string(),
  name: z.string(),
  input: z.record(z.string(), z.any()),
})

const anthropicMessageResponseSchema = z.object({
  id: z.string(),
  type: z.literal("message"),
  role: z.literal("assistant"),
  content: z.array(
    z.union([
      anthropicContentBlockTextSchema,
      anthropicContentBlockToolUseSchema,
    ]),
  ),
  model: z.string(),
  stop_reason: z.enum(["end_turn", "max_tokens", "stop_sequence", "tool_use"]),
  stop_sequence: z.string().nullable(),
  usage: anthropicUsageSchema,
})

/**
 * Validates if a response payload conforms to the Anthropic Message shape.
 * @param payload The response payload to validate.
 * @returns True if the payload is valid, false otherwise.
 */
function isValidAnthropicResponse(payload: unknown): boolean {
  return anthropicMessageResponseSchema.safeParse(payload).success
}

const anthropicStreamEventSchema = z.looseObject({
  type: z.enum([
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
  ]),
})

function isValidAnthropicStreamEvent(payload: unknown): boolean {
  return anthropicStreamEventSchema.safeParse(payload).success
}

/**
 * Builds a streaming chunk carrying GitHub Copilot's non-standard reasoning
 * fields (reasoning_text / reasoning_opaque) for thinking-translation tests.
 */
function makeReasoningChunk(
  id: string,
  reasoning: { reasoning_text?: string; reasoning_opaque?: string },
): ChatCompletionChunk {
  return {
    id,
    object: "chat.completion.chunk",
    created: 1677652288,
    model: "claude-opus-4.8",
    choices: [
      {
        index: 0,
        delta: { content: null, role: "assistant", ...reasoning },
        finish_reason: null,
        logprobs: null,
      },
    ],
  }
}

describe("OpenAI to Anthropic Non-Streaming Response Translation", () => {
  test("should translate a simple text response correctly", () => {
    const openAIResponse: ChatCompletionResponse = {
      id: "chatcmpl-123",
      object: "chat.completion",
      created: 1677652288,
      model: "gpt-4o-2024-05-13",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Hello! How can I help you today?",
          },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: 9,
        completion_tokens: 12,
        total_tokens: 21,
      },
    }

    const anthropicResponse = translateToAnthropic(openAIResponse)

    expect(isValidAnthropicResponse(anthropicResponse)).toBe(true)

    expect(anthropicResponse.id).toBe("chatcmpl-123")
    expect(anthropicResponse.stop_reason).toBe("end_turn")
    expect(anthropicResponse.usage.input_tokens).toBe(9)
    expect(anthropicResponse.content[0].type).toBe("text")
    if (anthropicResponse.content[0].type === "text") {
      expect(anthropicResponse.content[0].text).toBe(
        "Hello! How can I help you today?",
      )
    } else {
      throw new Error("Expected text block")
    }
  })

  test("should translate a response with tool calls", () => {
    const openAIResponse: ChatCompletionResponse = {
      id: "chatcmpl-456",
      object: "chat.completion",
      created: 1677652288,
      model: "gpt-4o-2024-05-13",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_abc",
                type: "function",
                function: {
                  name: "get_current_weather",
                  arguments: '{"location": "Boston, MA"}',
                },
              },
            ],
          },
          finish_reason: "tool_calls",
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: 30,
        completion_tokens: 20,
        total_tokens: 50,
      },
    }

    const anthropicResponse = translateToAnthropic(openAIResponse)

    expect(isValidAnthropicResponse(anthropicResponse)).toBe(true)

    expect(anthropicResponse.stop_reason).toBe("tool_use")
    expect(anthropicResponse.content[0].type).toBe("tool_use")
    if (anthropicResponse.content[0].type === "tool_use") {
      expect(anthropicResponse.content[0].id).toBe("call_abc")
      expect(anthropicResponse.content[0].name).toBe("get_current_weather")
      expect(anthropicResponse.content[0].input).toEqual({
        location: "Boston, MA",
      })
    } else {
      throw new Error("Expected tool_use block")
    }
  })

  test("should translate a response stopped due to length", () => {
    const openAIResponse: ChatCompletionResponse = {
      id: "chatcmpl-789",
      object: "chat.completion",
      created: 1677652288,
      model: "gpt-4o-2024-05-13",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "This is a very long response that was cut off...",
          },
          finish_reason: "length",
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 2048,
        total_tokens: 2058,
      },
    }

    const anthropicResponse = translateToAnthropic(openAIResponse)

    expect(isValidAnthropicResponse(anthropicResponse)).toBe(true)
    expect(anthropicResponse.stop_reason).toBe("max_tokens")
  })
})

describe("OpenAI to Anthropic Streaming Response Translation", () => {
  test("should translate a simple text stream correctly", () => {
    const openAIStream: Array<ChatCompletionChunk> = [
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: { role: "assistant" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: { content: "Hello" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: { content: " there" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-1",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          { index: 0, delta: {}, finish_reason: "stop", logprobs: null },
        ],
      },
    ]

    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      thinkingBlockOpen: false,
      thinkingRequested: false,
      toolCalls: {},
    }
    const translatedStream = openAIStream.flatMap((chunk) =>
      translateChunkToAnthropicEvents(chunk, streamState),
    )

    for (const event of translatedStream) {
      expect(isValidAnthropicStreamEvent(event)).toBe(true)
    }
  })

  test("should translate a stream with tool calls", () => {
    const openAIStream: Array<ChatCompletionChunk> = [
      {
        id: "cmpl-2",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: { role: "assistant" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-2",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_xyz",
                  type: "function",
                  function: { name: "get_weather", arguments: "" },
                },
              ],
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-2",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [{ index: 0, function: { arguments: '{"loc' } }],
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-2",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                { index: 0, function: { arguments: 'ation": "Paris"}' } },
              ],
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-2",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "gpt-4o-2024-05-13",
        choices: [
          { index: 0, delta: {}, finish_reason: "tool_calls", logprobs: null },
        ],
      },
    ]

    // Streaming translation requires state
    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      thinkingBlockOpen: false,
      thinkingRequested: false,
      toolCalls: {},
    }
    const translatedStream = openAIStream.flatMap((chunk) =>
      translateChunkToAnthropicEvents(chunk, streamState),
    )

    // These tests will fail until the stub is implemented
    for (const event of translatedStream) {
      expect(isValidAnthropicStreamEvent(event)).toBe(true)
    }
  })
})

describe("Copilot Reasoning to Anthropic Thinking Translation", () => {
  test("should translate Copilot reasoning into thinking blocks when thinking is enabled", () => {
    const openAIStream: Array<ChatCompletionChunk> = [
      makeReasoningChunk("cmpl-think", { reasoning_text: "Let me think" }),
      makeReasoningChunk("cmpl-think", { reasoning_text: " about this." }),
      makeReasoningChunk("cmpl-think", {
        reasoning_opaque: "c2lnbmF0dXJlLWJsb2I=",
      }),
      {
        id: "cmpl-think",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "claude-opus-4.8",
        choices: [
          {
            index: 0,
            delta: { content: "The answer is 42." },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-think",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "claude-opus-4.8",
        choices: [
          { index: 0, delta: {}, finish_reason: "stop", logprobs: null },
        ],
      },
    ]

    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      thinkingBlockOpen: false,
      thinkingRequested: true,
      toolCalls: {},
    }
    const events = openAIStream.flatMap((chunk) =>
      translateChunkToAnthropicEvents(chunk, streamState),
    )

    for (const event of events) {
      expect(isValidAnthropicStreamEvent(event)).toBe(true)
    }

    // A thinking block must open at index 0 before any text block.
    const thinkingStart = events.find(
      (e) => e.type === "content_block_start" && e.index === 0,
    )
    expect(thinkingStart).toBeDefined()
    if (thinkingStart?.type === "content_block_start") {
      expect(thinkingStart.content_block.type).toBe("thinking")
    }

    const thinkingDeltas = events.filter(
      (e) =>
        e.type === "content_block_delta" && e.delta.type === "thinking_delta",
    )
    expect(thinkingDeltas.length).toBe(2)

    const signatureDeltas = events.filter(
      (e) =>
        e.type === "content_block_delta" && e.delta.type === "signature_delta",
    )
    expect(signatureDeltas.length).toBe(1)

    // The thinking block must be closed before the text block opens at index 1.
    const textStart = events.find(
      (e) => e.type === "content_block_start" && e.index === 1,
    )
    expect(textStart).toBeDefined()
    if (textStart?.type === "content_block_start") {
      expect(textStart.content_block.type).toBe("text")
    }
    expect(
      events.some((e) => e.type === "content_block_stop" && e.index === 0),
    ).toBe(true)

    // No reasoning should leak as a ping when thinking was requested.
    expect(events.some((e) => e.type === "ping")).toBe(false)
  })

  test("should drop reasoning and emit pings when thinking is NOT enabled", () => {
    const openAIStream: Array<ChatCompletionChunk> = [
      makeReasoningChunk("cmpl-noop", {
        reasoning_text: "hidden reasoning one",
      }),
      makeReasoningChunk("cmpl-noop", {
        reasoning_text: "hidden reasoning two",
      }),
      {
        id: "cmpl-noop",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "claude-opus-4.8",
        choices: [
          {
            index: 0,
            delta: { content: "Final visible answer." },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "cmpl-noop",
        object: "chat.completion.chunk",
        created: 1677652288,
        model: "claude-opus-4.8",
        choices: [
          { index: 0, delta: {}, finish_reason: "stop", logprobs: null },
        ],
      },
    ]

    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      thinkingBlockOpen: false,
      thinkingRequested: false,
      toolCalls: {},
    }
    const events = openAIStream.flatMap((chunk) =>
      translateChunkToAnthropicEvents(chunk, streamState),
    )

    for (const event of events) {
      expect(isValidAnthropicStreamEvent(event)).toBe(true)
    }

    // Each reasoning chunk should yield a keepalive ping, no thinking block.
    expect(events.filter((e) => e.type === "ping").length).toBe(2)
    expect(events.some((e) => e.type === "content_block_start")).toBe(true)
    expect(
      events.some(
        (e) =>
          e.type === "content_block_start"
          && e.content_block.type === "thinking",
      ),
    ).toBe(false)

    // The visible text must still open at index 0 (reasoning consumed no index).
    const textStart = events.find((e) => e.type === "content_block_start")
    if (textStart?.type === "content_block_start") {
      expect(textStart.index).toBe(0)
      expect(textStart.content_block.type).toBe("text")
    }
  })
})
