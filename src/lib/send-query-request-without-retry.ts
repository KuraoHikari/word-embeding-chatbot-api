import NodeFormData from "form-data";
import ky from "ky";

import env from "@/env";

interface QueryParams {
  query: string;
  userId: string;
  chatbotId: string;
  modelType?: "word2vec" | "fasttext"; // default: "fasttext"
  pdfTitle: string;
  isProposedModel?: boolean;
  topK?: number;
  useGPT?: boolean; // default: false
  gptModel?: string; // default: "gpt-3.5-turbo"
  similarityThreshold?: number;
  includeRAGAS?: boolean; // default: false
  promptTemplate?: string | null;
  maxToken?: number; // default: 500, min: 100, max: 2000
  temperature?: number; // default: 0.7, min: 0.0, max: 1.0
}

export async function sendQueryRequestWithoutRetry(
  params: QueryParams,
  password: string,
): Promise<Response> {
  const controller = new AbortController();
  const nodeFormData = new NodeFormData();

  try {
    // Process form fields
    const fields = [
      { name: "query", value: params.query },
      { name: "userId", value: params.userId },
      { name: "chatbotId", value: params.chatbotId },
      { name: "modelType", value: params.modelType },
      { name: "pdfTitle", value: params.pdfTitle },
      { name: "topK", value: params.topK?.toString() || "5" },
      { name: "similarityThreshold", value: params.similarityThreshold?.toString() || "0.4" },
      { name: "useGPT", value: params.useGPT ? "true" : "false" },
      { name: "gptModel", value: params.gptModel || "gpt-3.5-turbo" },
      { name: "includeRAGAS", value: params.includeRAGAS ? "true" : "false" },
      { name: "promptTemplate", value: params.promptTemplate || "" },
      { name: "maxToken", value: params.maxToken?.toString() || "500" },
      { name: "temperature", value: params.temperature?.toString() || "0.3" },
    ];

    fields.forEach(field => nodeFormData.append(field.name, field.value));

    // Set headers
    const headers = {
      "X-API-Password": password,
      ...nodeFormData.getHeaders(),
    };

    let isProposedModel: "baseline-model" | "proposed-model" = "proposed-model";

    if (!params.isProposedModel) {
      isProposedModel = "baseline-model";
    }

    const response = await ky.post(`${env.PYTHON_SERVER_URL}/query/${isProposedModel}`, {
      headers,
      // ignore ts error
      // @ts-expect-error request body is FormData
      body: nodeFormData.getBuffer(),
      signal: controller.signal,
      timeout: false, // Disable timeout
      retry: 0, // Disable retry
    });

    if (!response.ok) {
      // Handle potential non-JSON responses and validate error shape
      let errorMessage = `HTTP ${response.status}`;

      try {
        const errorData: unknown = await response.json();

        // Check if errorData is an object with a message property
        if (typeof errorData === "object" && errorData !== null && "message" in errorData) {
          const message = (errorData as { message: unknown }).message;
          if (typeof message === "string") {
            errorMessage = message;
          }
        }
      }
      catch {
        throw new Error("Failed to parse error response as JSON");
      }

      throw new Error(errorMessage);
    }

    return response;
  }
  catch (error) {
    console.error("Query request failed:", error);
    throw error;
  }
}
