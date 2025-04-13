import { sign } from "hono/jwt";

import env from "@/env";

// Function to generate an access token
export async function generateAccessToken(userId: number): Promise<string> {
  const payload = {
    sub: userId,
    exp: Math.floor(Date.now() / 1000) + 60 * 60 * 24 * Number(env.ACCESS_TOKEN_EXPIRES_IN),
  };
  const accessToken = await sign(payload, env.ACCESS_TOKEN_SECRET);
  return accessToken;
}
