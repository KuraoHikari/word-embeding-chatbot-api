import { bodyLimit } from "hono/body-limit";
import * as HttpStatusCodes from "stoker/http-status-codes";

export function bodyLimitMiddleware(maxSize: number) {
  return bodyLimit({
    maxSize,
    onError: (c) => {
      return c.json(
        {
          message: `File size exceeds maximum limit of ${maxSize / (1024 * 1024)}MB`,
          status: HttpStatusCodes.REQUEST_TOO_LONG,
        },
        HttpStatusCodes.REQUEST_TOO_LONG,
      );
    },
  });
}
