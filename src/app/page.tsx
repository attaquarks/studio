'use client';

import { generateMriReport } from '@/ai/flows/generate-mri-report';
import { answerMriQuestion } from '@/ai/flows/answer-mri-question'; // Import VQA flow
import Header from '@/components/layout/header';
import MriUploader from '@/components/mri-uploader';
import ReportDisplay from '@/components/report-display';
import VqaDisplay from '@/components/vqa-display'; // Import VQA display component
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Input } from '@/components/ui/input'; // Import Input component
import { Label } from '@/components/ui/label'; // Import Label component
import { useState } from 'react';
import { Loader2, AlertTriangle, MessageCircleQuestion, FileText } from 'lucide-react'; // Import new icons
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'; // Import Card components

export default function Home() {
  const [mriImageDataUri, setMriImageDataUri] = useState<string | null>(null);
  const [clinicalSummary, setClinicalSummary] = useState<string | null>(null);
  const [vqaAnswer, setVqaAnswer] = useState<string | null>(null); // State for VQA answer
  const [question, setQuestion] = useState<string>(''); // State for VQA question
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isAnsweringQuestion, setIsAnsweringQuestion] = useState(false); // State for VQA loading
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (imageDataUri: string | null) => {
    setMriImageDataUri(imageDataUri);
    setClinicalSummary(null); // Clear report when image changes
    setVqaAnswer(null); // Clear VQA answer when image changes
    setError(null); // Clear errors
  };

  const handleGenerateReport = async () => {
    if (!mriImageDataUri) {
      setError("Please upload an MRI image first.");
      return;
    }

    setIsGeneratingReport(true);
    setError(null);
    setClinicalSummary(null); // Clear previous report
    setVqaAnswer(null); // Clear VQA answer

    try {
      const result = await generateMriReport({ mriImageDataUri });
      setClinicalSummary(result.clinicalSummary);
    } catch (err) {
      console.error("Error generating report:", err);
      setError("Failed to generate report. Please ensure the image is valid and try again.");
      setClinicalSummary(null);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleAnswerQuestion = async () => {
    if (!mriImageDataUri) {
      setError("Please upload an MRI image first.");
      return;
    }
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setIsAnsweringQuestion(true);
    setError(null);
    setVqaAnswer(null); // Clear previous answer
    setClinicalSummary(null); // Clear report

    try {
      const result = await answerMriQuestion({ mriImageDataUri, question });
      setVqaAnswer(result.answer);
    } catch (err) {
      console.error("Error answering question:", err);
      setError("Failed to answer question. Please try again.");
      setVqaAnswer(null);
    } finally {
      setIsAnsweringQuestion(false);
    }
  };

  const isProcessing = isGeneratingReport || isAnsweringQuestion;

  return (
    <div className="flex min-h-screen flex-col bg-secondary">
      <Header />
      <main className="container mx-auto flex flex-1 flex-col items-center gap-8 p-4 py-8 md:p-6 lg:p-8">
        <div className="grid w-full max-w-6xl grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Left Column: Uploader */}
          <div className="lg:col-span-1">
            <MriUploader onImageUpload={handleImageUpload} isGenerating={isProcessing} />
          </div>

          {/* Center Column: Actions and VQA Display */}
          <div className="flex flex-col gap-6 lg:col-span-1">
             {/* VQA Input Card */}
            <Card className="w-full shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircleQuestion className="h-5 w-5 text-primary" />
                  Ask a Question (VQA)
                </CardTitle>
                <CardDescription>Ask a specific question about the uploaded scan.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid w-full items-center gap-1.5">
                  <Label htmlFor="question">Your Question</Label>
                  <Input
                    id="question"
                    type="text"
                    placeholder="e.g., Is there evidence of a tumor?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={isProcessing}
                  />
                </div>
                <Button
                  onClick={handleAnswerQuestion}
                  disabled={!mriImageDataUri || !question.trim() || isProcessing}
                  className="w-full"
                >
                  {isAnsweringQuestion ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Answering...
                    </>
                  ) : (
                    'Ask Question'
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Generate Report Card */}
             <Card className="w-full shadow-md">
               <CardHeader>
                 <CardTitle className="flex items-center gap-2">
                   <FileText className="h-5 w-5 text-primary" />
                    Generate Report
                 </CardTitle>
                 <CardDescription>Generate a general clinical summary.</CardDescription>
               </CardHeader>
               <CardContent>
                 <Button
                   onClick={handleGenerateReport}
                   disabled={!mriImageDataUri || isProcessing}
                   className="w-full"
                 >
                   {isGeneratingReport ? (
                     <>
                       <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                       Generating Report...
                     </>
                   ) : (
                     'Generate Report'
                   )}
                 </Button>
               </CardContent>
             </Card>

          </div>

           {/* Right Column: Report and VQA Answer Display */}
          <div className="flex flex-col gap-6 lg:col-span-1">
             <ReportDisplay report={clinicalSummary} isGenerating={isGeneratingReport} />
             <VqaDisplay answer={vqaAnswer} isGenerating={isAnsweringQuestion} />
           </div>

        </div>

        {error && (
           <Alert variant="destructive" className="w-full max-w-6xl">
             <AlertTriangle className="h-4 w-4" />
             <AlertTitle>Error</AlertTitle>
             <AlertDescription>{error}</AlertDescription>
           </Alert>
         )}

      </main>
      <footer className="py-4 text-center text-sm text-muted-foreground">
         NeuroReport © {new Date().getFullYear()}
      </footer>
    </div>
  );
}
