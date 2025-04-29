import type { Options as KyOptions } from "ky";

import NodeFormData from "form-data";
import ky from "ky";

import env from "@/env";

interface TrainingRequestOptions extends KyOptions {
  retries?: number;
  initialDelay?: number;
  timeout?: number;
}

export async function sendTrainingRequestWithRetry(
  form: FormData,
  options: TrainingRequestOptions = {},
): Promise<Response> {
  const {
    retries = 3,
    initialDelay = 1000,
    timeout = 30000,
    ..._kyOptions
  } = options as { [key: string]: any };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    return await ky.post(`${env.PYTHON_SERVER_URL}/train`, {
      body: Object.assign(new NodeFormData(), form),
      signal: controller.signal,
      retry: {
        limit: retries,
        methods: ["post"],
        statusCodes: [408, 413, 429, 500, 502, 503, 504],
        backoffLimit: 10000, // Max delay 10 detik
      },
      hooks: {
        beforeRetry: [
          async ({ request: _request, retryCount }) => {
            const delay = Math.min(initialDelay * (2 ** retryCount), 10000);

            await new Promise(resolve => setTimeout(resolve, delay));
          },
        ],
      },
    });
  }
  finally {
    clearTimeout(timeoutId);
  }
}
