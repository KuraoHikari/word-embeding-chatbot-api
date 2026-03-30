import type { QueryBaselineModelResponse, QueryProposedModelResponse } from "@/lib/types";

import db from "@/db";
import { messages, modelResponses } from "@/db/schema";
import env from "@/env";
import { sendQueryRequestWithoutRetry } from "@/lib/send-query-request-without-retry";

export interface SaveMessageParams {
 userId: number;
 conversationId: number;
 text: string;
 isBot: boolean;
 senderRole: "admin" | "bot" | "contact";
}

export interface SendAndSaveBotReplyParams {
 chatbot: {
  id: number;
  userId: number;
  embeddingModel: string;
  systemPrompt: string;
  pdfTitle: string;
  aiModel: string;
  maxTokens: number;
  temperature: number;
  isProposedModel: boolean;
 };
 conversationId: number;
 userMessageText: string;
}

/**
 * Persist a single message row and return it.
 */
export async function saveMessage(params: SaveMessageParams) {
 const [saved] = await db
  .insert(messages)
  .values({
   userId: params.userId,
   conversationId: params.conversationId,
   text: params.text,
   isBot: params.isBot,
   senderRole: params.senderRole,
  })
  .returning();

 return saved;
}

/**
 * Query the ML backend, persist the bot answer + model response, and return both.
 */
export async function sendAndSaveBotReply(params: SendAndSaveBotReplyParams) {
 const { chatbot, conversationId, userMessageText } = params;

 const response = await sendQueryRequestWithoutRetry(
  {
   query: userMessageText,
   userId: String(chatbot.userId),
   chatbotId: String(chatbot.id),
   modelType: chatbot.embeddingModel === "fasttext" || chatbot.embeddingModel === "word2vec" ? chatbot.embeddingModel : "fasttext",
   promptTemplate: chatbot.systemPrompt || "",
   pdfTitle: chatbot.pdfTitle,
   topK: 3,
   useGPT: true,
   gptModel: chatbot.aiModel || "gpt-3.5-turbo",
   similarityThreshold: 0.4,
   includeRAGAS: true,
   maxToken: chatbot.maxTokens || 500,
   temperature: chatbot.temperature ? chatbot.temperature / 100 : 0.3,
   isProposedModel: chatbot.isProposedModel || false,
  },
  env.API_PASSWORD,
 );

 if (!response.ok) {
  throw new Error("ML backend returned a non-OK response");
 }

 const results: QueryProposedModelResponse | QueryBaselineModelResponse = chatbot.isProposedModel
  ? ((await response.json()) as QueryProposedModelResponse)
  : ((await response.json()) as QueryBaselineModelResponse);

 const botText = chatbot.isProposedModel
  ? (results as QueryProposedModelResponse).gpt_generation?.answer || "Maaf, saya tidak dapat menjawab pertanyaan Anda saat ini."
  : (results as QueryBaselineModelResponse).gpt_generation?.answer || "Maaf, saya tidak dapat menjawab pertanyaan Anda saat ini.";

 const botAnswer = await saveMessage({
  userId: chatbot.userId,
  conversationId,
  text: botText,
  isBot: true,
  senderRole: "bot",
 });

 // Persist model response metadata
 const modelResponseData = chatbot.isProposedModel
  ? {
     messageId: botAnswer.id,
     modelType: "proposed" as const,
     query: results.query,
     processingTime: Math.round(results.processing_time),
     results: results.results,
     metadata: results.metadata,
     complexityAnalysis: (results as QueryProposedModelResponse).complexity_analysis,
     searchPipeline: (results as QueryProposedModelResponse).search_pipeline,
     gptGeneration: results.gpt_generation || null,
     ragasEvaluation: results.ragas_evaluation || null,
     message: (results as QueryProposedModelResponse).message || null,
     userId: chatbot.userId,
     chatbotId: chatbot.id,
    }
  : {
     messageId: botAnswer.id,
     modelType: "baseline" as const,
     query: results.query,
     processingTime: Math.round(results.processing_time),
     results: results.results,
     metadata: results.metadata,
     modelApproach: (results as QueryBaselineModelResponse).model_approach,
     pipelineSteps: (results as QueryBaselineModelResponse).pipeline_steps,
     gptGeneration: (results as QueryBaselineModelResponse).gpt_generation || null,
     ragasEvaluation: results.ragas_evaluation || null,
     userId: chatbot.userId,
     chatbotId: chatbot.id,
    };

 await db.insert(modelResponses).values(modelResponseData);

 return botAnswer;
}
