import type { OpenAPIHono, RouteConfig, RouteHandler } from "@hono/zod-openapi";
import type { Schema } from "hono";
import type { PinoLogger } from "hono-pino";

export interface AppBindings {
  Variables: {
    logger: PinoLogger;
    userId?: number; // <- ditambahkan
    contactId?: number; // <- ditambahkan
  };
}

// eslint-disable-next-line ts/no-empty-object-type
export type AppOpenAPI<S extends Schema = {}> = OpenAPIHono<AppBindings, S>;

export type AppRouteHandler<R extends RouteConfig> = RouteHandler<R, AppBindings>;

export interface QueryProposedModelResponse {
  status: "success";
  query: string;
  processing_time: number;
  complexity_analysis: {
    type: string;
    score: number;
    word_count: number;
    unique_words: number;
    question_words: number;
    weights_used: Record<string, number>;
  };
  search_pipeline: {
    hybrid_search_results: number;
    mmr_reranked_results: number;
    mmr_lambda: number;
    similarity_threshold: number;
  };
  results: Array<{
    rank: number;
    text: string;
    doc_index: number;
    final_score: number;
    diversity_penalty: number;
    original_rank: number;
    detailed_scores?: {
      fasttext_similarity: number;
      bm25_score: number;
      context_score: number;
      weighted_score: number;
    };
    context_range?: string;
  }>;
  metadata: {
    model_type: string;
    documents_count: number;
    features_used: {
      semantic_search: boolean;
      keyword_search: boolean;
      context_scoring: boolean;
      mmr_reranking: boolean;
      gpt_generation: boolean;
      ragas_evaluation: boolean;
    };
  };
  gpt_generation?: {
    answer?: string;
    model_used?: string;
    tokens_used?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    error?: string;
  };
  ragas_evaluation?: {
    context_relevance?: number;
    faithfulness?: number;
    answer_relevance?: number;
    overall_score?: number;
    error?: string;
  };
  message?: string;
};

export interface QueryBaselineModelResponse {
  status: "success";
  query: string;
  processing_time: number;
  model_approach: string;
  pipeline_steps: string[];
  results: Array<{
    rank: number;
    text: string;
    similarity_score: number;
    doc_index: number;
    method: string;
  }>;
  metadata: {
    model_type: string;
    documents_count: number;
    embedding_dimension: number;
    hyperparameters: Record<string, any>;
    features_used: {
      semantic_search: boolean;
      keyword_search: boolean;
      context_scoring: boolean;
      mmr_reranking: boolean;
      gpt_generation: boolean;
      ragas_evaluation: boolean;
    };
  };
  gpt_generation?: {
    answer?: string;
    model_used?: string;
    tokens_used?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    error?: string;
  };
  ragas_evaluation?: {
    context_relevance?: number;
    faithfulness?: number;
    answer_relevance?: number;
    overall_score?: number;
    error?: string;
  };
};
