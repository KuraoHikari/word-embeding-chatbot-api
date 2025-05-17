import { createRoute, z } from "@hono/zod-openapi";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { IdParamsSchema } from "stoker/openapi/schemas";

import { insertConversationsSchema, selectConversationsSchema } from "@/db/schema";

const tags = ["Conversations"];

export const list = createRoute({
  path: "/conversations",
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.array(selectConversationsSchema),
      "List of conversations",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
  },
});

export const create = createRoute({
  path: "/conversations",
  method: "post",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
    body: jsonContentRequired(insertConversationsSchema, "The conversation to create"),
  },
  responses: {
    [HttpStatusCodes.CREATED]: jsonContent(
      selectConversationsSchema,
      "Conversation created",
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

export const getOne = createRoute({
  path: "/conversations/{id}",
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
    params: IdParamsSchema,
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      selectConversationsSchema,
      "Conversation found",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Conversation not found",
    ),

  },
});

export const remove = createRoute({
  path: "/conversations/{id}",
  method: "delete",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
    params: IdParamsSchema,
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({ message: z.string() }),
      "Conversation deleted",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Conversation not found",
    ),
    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.INTERNAL_SERVER_ERROR,
    ),
  },
});

export type ListRoute = typeof list;
export type CreateRoute = typeof create;
export type GetOneRoute = typeof getOne;
export type RemoveRoute = typeof remove;
