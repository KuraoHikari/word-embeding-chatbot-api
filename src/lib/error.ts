import { HTTPException } from "hono/http-exception";
import * as HttpStatusCodes from "stoker/http-status-codes";

export function TooManyRequests(message: string = "Too many requests") {
  return new HTTPException(HttpStatusCodes.TOO_MANY_REQUESTS, { message });
}

export function Forbidden(message: string = "Forbidden") {
  return new HTTPException(HttpStatusCodes.FORBIDDEN, { message });
}

export function Unauthorized(message: string = "Unauthorized") {
  return new HTTPException(HttpStatusCodes.UNAUTHORIZED, { message });
}

export function NotFound(message: string = "Not Found") {
  return new HTTPException(HttpStatusCodes.NOT_FOUND, { message });
}

export function BadRequest(message: string = "Bad Request") {
  return new HTTPException(HttpStatusCodes.BAD_REQUEST, { message });
}

export function InternalError(message: string = "Internal Error") {
  return new HTTPException(HttpStatusCodes.INTERNAL_SERVER_ERROR, { message });
}
