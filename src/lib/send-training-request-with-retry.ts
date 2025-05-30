import type { Options as KyOptions } from "ky";

import NodeFormData from "form-data";
import ky from "ky";
import { Buffer } from "node:buffer";

import env from "@/env";

interface TrainingRequestOptions extends KyOptions {
  retries?: number;
  initialDelay?: number;
  timeout?: number;
}

export async function sendTrainingRequestWithRetry(
  form: FormData,
  options: TrainingRequestOptions = {},
  password: string,
): Promise<Response> {
  const {
    retries = 3,
    initialDelay = 1000,
    timeout = 120000, // 2 menit
    ..._kyOptions
  } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const nodeFormData = new NodeFormData();

    // Process form fields
    const fields = [
      { name: "userId", value: form.get("userId")?.toString() || "" },
      { name: "chatbotId", value: form.get("chatbotId")?.toString() || "" },
      { name: "modelType", value: form.get("modelType")?.toString() || "" },
      { name: "pdfTitle", value: form.get("pdfTitle")?.toString() || "" },
    ];

    fields.forEach(field => nodeFormData.append(field.name, field.value));

    // Process PDF file
    const pdfFile = form.get("pdf") as File;
    if (pdfFile) {
      const buffer = await pdfFile.arrayBuffer();
      nodeFormData.append("pdf", Buffer.from(buffer), {
        filename: pdfFile.name,
        contentType: pdfFile.type,
        knownLength: buffer.byteLength,
      });
    }

    // Set headers
    const headers = {
      ..._kyOptions.headers,
      "X-API-Password": password,
      ...nodeFormData.getHeaders(),
    };

    const response = await ky.post(`${env.PYTHON_SERVER_URL}/train`, {
      ..._kyOptions,
      headers,
      body: nodeFormData.getBuffer(),
      signal: controller.signal,
      retry: {
        limit: retries,
        methods: ["post"],
        statusCodes: [408, 413, 429, 500, 502, 503, 504],
        backoffLimit: 120000, // Maksimal backoff 2 menit
      },
      hooks: {
        beforeRetry: [
          async ({ retryCount }) => {
            const delay = Math.min(initialDelay * (2 ** retryCount), 120000);
            console.log(`Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
          },
        ],
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error?.message || `HTTP ${response.status}`);
    }

    return response;
  }
  catch (error) {
    console.error("Training request failed:", error);
    throw error;
  }
  finally {
    clearTimeout(timeoutId);
  }
}
