'use server';

/**
 * @fileOverview Answers questions about an MRI scan based on the image and question provided.
 *
 * - answerMriQuestion - A function that handles the VQA process.
 * - AnswerMriQuestionInput - The input type for the answerMriQuestion function.
 * - AnswerMriQuestionOutput - The return type for the answerMriQuestion function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const AnswerMriQuestionInputSchema = z.object({
  mriImageDataUri: z
    .string()
    .describe(
      "An MRI image, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  question: z.string().describe('The specific question asked about the MRI scan.'),
});
export type AnswerMriQuestionInput = z.infer<typeof AnswerMriQuestionInputSchema>;

const AnswerMriQuestionOutputSchema = z.object({
  answer: z.string().describe('The answer to the question based on the MRI scan analysis.'),
});
export type AnswerMriQuestionOutput = z.infer<typeof AnswerMriQuestionOutputSchema>;

export async function answerMriQuestion(input: AnswerMriQuestionInput): Promise<AnswerMriQuestionOutput> {
  return answerMriQuestionFlow(input);
}

const prompt = ai.definePrompt({
  name: 'answerMriQuestionPrompt',
  input: {
    schema: AnswerMriQuestionInputSchema,
  },
  output: {
    schema: AnswerMriQuestionOutputSchema,
  },
  prompt: `You are a medical expert specializing in radiology. Analyze the provided MRI image and answer the specific question asked about it. Provide a concise and accurate answer based *only* on the visual information in the image.

MRI Image:
{{media url=mriImageDataUri}}

Question: {{{question}}}

Answer:`,
});

const answerMriQuestionFlow = ai.defineFlow<
  typeof AnswerMriQuestionInputSchema,
  typeof AnswerMriQuestionOutputSchema
>(
  {
    name: 'answerMriQuestionFlow',
    inputSchema: AnswerMriQuestionInputSchema,
    outputSchema: AnswerMriQuestionOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
