import { createRoute, z } from "@hono/zod-openapi";
import { title } from "node:process";
import * as HttpStatusCodes from "stoker/http-status-codes";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { createErrorSchema, IdParamsSchema } from "stoker/openapi/schemas";

import { insertChatbotsSchema, selectChatbotsSchema } from "@/db/schema";
import { notFoundSchema } from "@/lib/constants";

const tags = ["Chatbots"];

const CreateChatbotFormSchema = z.object({
  pdf: z.instanceof(File).or(z.instanceof(Blob)).optional().describe("PDF file upload").openapi({ format: "binary" }),
  title: z.string().min(1).max(255),
  description: z.string().max(1000).optional(),
  commandTemplate: z.string().min(1),
  modelAi: z.enum(["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-4o"]).describe("OpenAI model to use"),
  embedingModel: z.enum(["word2vec", "fasttext", "pinecone"]).describe("Embedding model type"),
  sugestionMessage: z.string().min(1),
//   pdfTitle: z.string().min(1),
});

export const list = createRoute({
  path: "/chatbots",
  method: "get",
  tags,
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
      "Invalid request",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      "Internal server error",
    ),
  },
});

export type ListChatbot = typeof list;

export type CreateChatbot = typeof create;
