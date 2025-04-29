import { Document as LangchainDocument, RecursiveCharacterTextSplitter } from "@pinecone-database/doc-splitter";
import { Pinecone } from "@pinecone-database/pinecone";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import md5 from "md5";
import { Configuration, OpenAIApi } from "openai-edge";

import env from "@/env";

interface LoadPDFParams {
  pdfBlob: Blob;
  namespace: string;
}

// Type definitions
interface PageMetadata {
  loc: {
    pageNumber: number;
  };
}

interface PageDocument {
  pageContent: string;
  metadata: PageMetadata;
}

interface SplitDocumentMetadata {
  text: string;
  pageNumber: number;
}

interface VectorMetadata extends SplitDocumentMetadata {
  [key: string]: any;
}

interface PineconeVector {
  id: string;
  values: number[];
  metadata: VectorMetadata;
}

interface PineconeMatch {
  id: string;
  score: number;
  metadata: VectorMetadata;
}

// Pinecone client initialization
export function getPineconeClient(): Pinecone {
  return new Pinecone({
    apiKey: env.PINECONE_API_KEY!,
  });
}

export async function loadPDFIntoPinecone(
  { pdfBlob, namespace }: LoadPDFParams,
): Promise<void> {
  try {
    const loader = new PDFLoader(pdfBlob);
    const rawPages = await loader.load();
    const pages: PageDocument[] = rawPages.map(page => ({
      pageContent: page.pageContent,
      metadata: {
        loc: {
          pageNumber: page.metadata.pageNumber,
        },
      },
    }));

    const documents = await Promise.all(pages.map(prepareDocument));
    const vectors = await Promise.all(documents.flat().map(embedDocument));

    const client = getPineconeClient();

    const pineconeIndex = client.index("chatpdf");
    const pineconeNamespace = pineconeIndex.namespace(convertToAscii(namespace));

    await pineconeNamespace.upsert(vectors);
  }
  catch (error) {
    console.error("Failed to load PDF to Pinecone:", error);
    throw new Error("PDF processing failed");
  }
}

// Document preparation
async function prepareDocument(page: PageDocument): Promise<LangchainDocument[]> {
  let { pageContent, metadata } = page;
  pageContent = pageContent.replace(/\n/g, "");

  const splitter = new RecursiveCharacterTextSplitter();
  const docs = await splitter.splitDocuments([
    new LangchainDocument({
      pageContent,
      metadata: {
        pageNumber: metadata.loc.pageNumber,
        text: truncateStringByBytes(pageContent, 36000),
      },
    }),
  ]);
  return docs;
}

// Document embedding
async function embedDocument(doc: LangchainDocument): Promise<PineconeVector> {
  const embeddings = await getEmbeddings(doc.pageContent);
  const hash = md5(doc.pageContent);

  return {
    id: hash,
    values: embeddings,
    metadata: {
      text: doc.metadata.text as string,
      pageNumber: doc.metadata.pageNumber as number,
    },
  };
}

// Utility functions
export function truncateStringByBytes(str: string, bytes: number): string {
  const enc = new TextEncoder();
  return new TextDecoder("utf-8").decode(enc.encode(str).slice(0, bytes));
}

export function convertToAscii(inputString: string): string {
  // eslint-disable-next-line no-control-regex
  return inputString.replace(/[^\x00-\x7F]+/g, "");
}

// OpenAI embeddings
const config = new Configuration({
  apiKey: env.OPEN_AI_API_KEY!,
});

const openai = new OpenAIApi(config);

export async function getEmbeddings(text: string): Promise<number[]> {
  const response = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: text.replace(/\n/g, " "),
  });
  const result = await response.json();
  return result.data[0].embedding;
}

// Query functions
export async function getMatchesFromEmbeddings(
  embeddings: number[],
  fileKey: string,
): Promise<PineconeMatch[]> {
  const client = getPineconeClient();
  const pineconeIndex = client.index("chatpdf");
  const namespace = pineconeIndex.namespace(convertToAscii(fileKey));

  const queryResult = await namespace.query({
    topK: 5,
    vector: embeddings,
    includeMetadata: true,
  });

  return queryResult.matches.map(match => ({
    id: match.id,
    score: match.score,
    metadata: {
      ...match.metadata,
      text: match.metadata?.text || "",
      pageNumber: match.metadata?.pageNumber || 0,
    },
  })) as PineconeMatch[];
}
