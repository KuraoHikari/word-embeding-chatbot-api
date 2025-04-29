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
  modelAi: z.string().min(1),
  embedingModel: z.string().min(1),
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
    // 401

    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
  },
});

export type ListChatbot = typeof list;

export type CreateChatbot = typeof create;
