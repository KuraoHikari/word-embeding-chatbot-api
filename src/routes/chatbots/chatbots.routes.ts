import { createRoute, z } from "@hono/zod-openapi";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { createErrorSchema, IdParamsSchema } from "stoker/openapi/schemas";

import { patchChatbotsSchema, selectChatbotsSchema } from "@/db/schema";

const tags = ["Chatbots"];

const CreateChatbotFormSchema = z.object({
  pdf: z.instanceof(File).or(z.instanceof(Blob)).optional().describe("PDF file upload").openapi({ format: "binary" }),
  title: z.string().min(1).max(255),
  description: z.string().max(1000).optional(),
  commandTemplate: z.string().min(1),
  modelAi: z.enum(["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4o"]).describe("OpenAI model to use"),
  embedingModel: z.enum(["word2vec", "fasttext", "pinecone"]).describe("Embedding model type"),
  sugestionMessage: z.string().min(1),
  modelType: z.enum(["proposed-model", "baseline-model"]).describe("Type of model"),
});

export const list = createRoute({
  path: "/chatbots",
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.array(selectChatbotsSchema),
      "List of chatbots",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),

  },
});

export const create = createRoute({
  path: "/chatbots",
  method: "post",
  tags,
  request: {
    body: {
      content: {
        "multipart/form-data": {
          schema: CreateChatbotFormSchema,

        },
      },
      required: true,
    },
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.CREATED]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot created successfully",
    ),
    [HttpStatusCodes.BAD_REQUEST]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.BAD_REQUEST,
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.UNAUTHORIZED,
    ),
    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.INTERNAL_SERVER_ERROR,
    ),
    [HttpStatusCodes.REQUEST_TOO_LONG]: jsonContent(
      z.object({ message: z.string() }),
      "File size exceeds limit",
    ),
    [HttpStatusCodes.TOO_MANY_REQUESTS]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.TOO_MANY_REQUESTS,
    ),
  },
});

export const getOne = createRoute({
  path: "/chatbots/{id}",
  method: "get",
  tags,
  request: {
    params: IdParamsSchema,
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot retrieved successfully",
    ),
    [HttpStatusCodes.BAD_REQUEST]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.BAD_REQUEST,
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.UNAUTHORIZED,
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot not found",
    ),
    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.INTERNAL_SERVER_ERROR,
    ),
  },
});

export const remove = createRoute({
  path: "/chatbots/{id}",
  method: "delete",
  tags,
  request: {
    params: IdParamsSchema,
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot deleted successfully",
    ),
    [HttpStatusCodes.BAD_REQUEST]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.BAD_REQUEST,
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.UNAUTHORIZED,
    ),
    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.INTERNAL_SERVER_ERROR,
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot not found",
    ),
  },

});

export const patch = createRoute({
  path: "/chatbots/{id}",
  method: "patch",
  tags,
  request: {
    params: IdParamsSchema,
    body: jsonContentRequired(
      patchChatbotsSchema,
      "The chatbot updates",
    ),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot updated successfully",
    ),
    [HttpStatusCodes.BAD_REQUEST]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.BAD_REQUEST,
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.UNAUTHORIZED,
    ),
    [HttpStatusCodes.UNPROCESSABLE_ENTITY]: jsonContent(
      createErrorSchema(patchChatbotsSchema)
        .or(createErrorSchema(IdParamsSchema)),
      "The validation error(s)",
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot not found",
    ),
    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.INTERNAL_SERVER_ERROR,
    ),
  },
});

export type ListChatbot = typeof list;

export type CreateChatbot = typeof create;

export type GetChatbot = typeof getOne;

export type DeleteChatbot = typeof remove;

export type PatchChatbot = typeof patch;
