import * as argon2 from "argon2";

export async function hash(data: string): Promise<string> {
  return argon2.hash(data);
}

export async function verify(hash: string, data: string): Promise<boolean> {
  return argon2.verify(hash, data);
}
