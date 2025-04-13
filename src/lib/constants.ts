import * as HttpStatusPhrases from "stoker/http-status-phrases";
import { createMessageObjectSchema } from "stoker/openapi/schemas";

export const ZOD_ERROR_MESSAGES = {
  REQUIRED: "Required",
  EXPECTED_NUMBER: "Expected number, received nan",
  NO_UPDATES: "No updates provided",
};

export const ZOD_ERROR_CODES = {
  INVALID_UPDATES: "invalid_updates",
};

export const notFoundSchema = createMessageObjectSchema(HttpStatusPhrases.NOT_FOUND);

export const createdSchema = createMessageObjectSchema(HttpStatusPhrases.CREATED);
export const noContentSchema = createMessageObjectSchema(HttpStatusPhrases.NO_CONTENT);
export const okSchema = createMessageObjectSchema(HttpStatusPhrases.OK);
export const badRequestSchema = createMessageObjectSchema(HttpStatusPhrases.BAD_REQUEST);
export const unauthorizedSchema = createMessageObjectSchema(HttpStatusPhrases.UNAUTHORIZED);
export const forbiddenSchema = createMessageObjectSchema(HttpStatusPhrases.FORBIDDEN);
export const conflictSchema = createMessageObjectSchema(HttpStatusPhrases.CONFLICT);
export const unprocessableEntitySchema = createMessageObjectSchema(
  HttpStatusPhrases.UNPROCESSABLE_ENTITY,
);
export const internalServerErrorSchema = createMessageObjectSchema(
  HttpStatusPhrases.INTERNAL_SERVER_ERROR,
);
export const notImplementedSchema = createMessageObjectSchema(
  HttpStatusPhrases.NOT_IMPLEMENTED,
);
export const serviceUnavailableSchema = createMessageObjectSchema(
  HttpStatusPhrases.SERVICE_UNAVAILABLE,
);
export const gatewayTimeoutSchema = createMessageObjectSchema(
  HttpStatusPhrases.GATEWAY_TIMEOUT,
);
export const badGatewaySchema = createMessageObjectSchema(
  HttpStatusPhrases.BAD_GATEWAY,
);
export const tooManyRequestsSchema = createMessageObjectSchema(
  HttpStatusPhrases.TOO_MANY_REQUESTS,
);
export const preconditionFailedSchema = createMessageObjectSchema(
  HttpStatusPhrases.PRECONDITION_FAILED,
);
export const notAcceptableSchema = createMessageObjectSchema(
  HttpStatusPhrases.NOT_ACCEPTABLE,
);

export const requestTimeoutSchema = createMessageObjectSchema(
  HttpStatusPhrases.REQUEST_TIMEOUT,
);
export const lengthRequiredSchema = createMessageObjectSchema(
  HttpStatusPhrases.LENGTH_REQUIRED,
);
export const methodNotAllowedSchema = createMessageObjectSchema(
  HttpStatusPhrases.METHOD_NOT_ALLOWED,
);
