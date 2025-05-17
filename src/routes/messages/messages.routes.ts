import { createRoute, z } from "@hono/zod-openapi";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { IdParamsSchema } from "stoker/openapi/schemas";

import { insertMessagesSchema, selectMessagesSchema } from "@/db/schema";

const tags = ["Messages"];

export const list = createRoute({
  path: "/messages",
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.array(selectMessagesSchema),
      "List of messages",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
  },
});

export const create = createRoute({
  path: "/messages",
  method: "post",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
    body: jsonContentRequired(insertMessagesSchema, "The message to create"),
  },
  responses: {
    [HttpStatusCodes.CREATED]: jsonContent(
      selectMessagesSchema,
      "Message created",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot not found",
    ),
  },
});

export type ListRoute = typeof list;
export type CreateRoute = typeof create;
