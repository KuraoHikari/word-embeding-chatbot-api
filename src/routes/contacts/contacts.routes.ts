import { createRoute, z } from "@hono/zod-openapi";
import * as HttpStatusCodes from "stoker/http-status-codes";
import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { jsonContent, jsonContentRequired } from "stoker/openapi/helpers";
import { createErrorSchema, IdParamsSchema } from "stoker/openapi/schemas";

import { createContactSchema, patchContactSchema, selectContactSchema } from "@/db/schema";

const tags = ["Contacts"];

export const list = createRoute({
  path: "/contacts",
  method: "get",
  tags,
  request: {
    headers: z.object({
      authorization: z.string().optional(),
    }),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.array(selectContactSchema),
      "List of contacts",
    ),
    [HttpStatusCodes.UNAUTHORIZED]: jsonContent(
      z.object({ message: z.string() }),
      "Unauthorized",
    ),
  },
});

export const create = createRoute({
  path: "/contacts",
  method: "post",
  tags,
  request: {
    body: jsonContentRequired(createContactSchema.extend({
      chatbotId: z.number(),
    }), "The contact to create"),
  },
  responses: {
    [HttpStatusCodes.OK]: jsonContent(
      z.object({
        access_token: z.string(),
      }),
      "Contact created",
    ),
    // not found
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot not found",
    ),

    [HttpStatusCodes.INTERNAL_SERVER_ERROR]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.INTERNAL_SERVER_ERROR,
    ),
    [HttpStatusCodes.REQUEST_TOO_LONG]: jsonContent(
      z.object({ message: z.string() }),
      HttpStatusPhrases.REQUEST_TOO_LONG,
    ),
  },
});

export const getOne = createRoute({
  path: "/contacts/{id}",
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
      selectContactSchema,
      "Contact",
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Chatbot not found",
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
  },
});

export const remove = createRoute({
  path: "/contacts/{id}",
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
      "Contact deleted",
    ),
    [HttpStatusCodes.NOT_FOUND]: jsonContent(
      z.object({ message: z.string() }),
      "Contact not found",
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
  },
});

export type ListRoute = typeof list;
export type CreateRoute = typeof create;
export type GetOneRoute = typeof getOne;
export type RemoveRoute = typeof remove;
