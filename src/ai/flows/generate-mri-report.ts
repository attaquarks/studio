'use server';

/**
 * @fileOverview Generates a clinical summary of MRI findings from an uploaded image.
 *
 * - generateMriReport - A function that handles the MRI report generation process.
 * - GenerateMriReportInput - The input type for the generateMriReport function.
 * - GenerateMriReportOutput - The return type for the generateMriReport function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const GenerateMriReportInputSchema = z.object({
  mriImageDataUri: z
    .string()
    .describe(
      "An MRI image, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type GenerateMriReportInput = z.infer<typeof GenerateMriReportInputSchema>;

const GenerateMriReportOutputSchema = z.object({
  clinicalSummary: z.string().describe('A clinical summary of the MRI findings.'),
});
export type GenerateMriReportOutput = z.infer<typeof GenerateMriReportOutputSchema>;

export async function generateMriReport(input: GenerateMriReportInput): Promise<GenerateMriReportOutput> {
  return generateMriReportFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateMriReportPrompt',
  input: {
    schema: z.object({
      mriImageDataUri: z
        .string()
        .describe(
          "An MRI image, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
        ),
    }),
  },
  output: {
    schema: z.object({
      clinicalSummary: z.string().describe('A clinical summary of the MRI findings.'),
    }),
  },
  prompt: `You are a medical expert specializing in radiology. Analyze the MRI image and provide a clinical summary of the findings.

Here is the MRI image:

{{media url=mriImageDataUri}}`,
});

const generateMriReportFlow = ai.defineFlow<
  typeof GenerateMriReportInputSchema,
  typeof GenerateMriReportOutputSchema
>(
  {
    name: 'generateMriReportFlow',
    inputSchema: GenerateMriReportInputSchema,
    outputSchema: GenerateMriReportOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
