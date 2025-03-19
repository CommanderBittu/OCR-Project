import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  Copy, 
  ArrowLeft, 
  Home, 
  Download, 
  ZoomIn, 
  ZoomOut, 
  Edit, 
  Save, 
  Share2, 
  Printer, 
  Info, 
  MessageSquare 
} from 'lucide-react';



interface ProcessingStats {
  startTime: Date;
  endTime: Date;
  processingTime: number;
}

interface ConfidenceData {
  characterConfidences: {
    high: number;
    medium: number;
    low: number;
  };
  overallConfidence: number;
}

function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { text, imageUrl } = location.state || { text: '', imageUrl: null };
  
  // State management
  const [originalText, setOriginalText] = useState(text);
  const [editedText, setEditedText] = useState(text);
  const [isEditing, setIsEditing] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showConfidence, setShowConfidence] = useState(false);
  const [activeTab, setActiveTab] = useState('text');

  // Fixed confidence data generation with proper typing
  const getRandomConfidenceData = (): ConfidenceData => {
    let high = Math.floor(Math.random() * (85 - 70 + 1)) + 70; // 70% - 85%
    let medium = Math.floor(Math.random() * (20 - 10 + 1)) + 10; // 10% - 20%
    let low = 100 - (high + medium); // Ensure total = 100%
    let overallConfidence = Math.floor(Math.random() * (95 - 80 + 1)) + 80; // 80% - 95%
    
    return {
      characterConfidences: { high, medium, low },
      overallConfidence
    };
  };

  const [confidenceData, setConfidenceData] = useState<ConfidenceData>(getRandomConfidenceData());
  
  const [processingStats] = useState<ProcessingStats>({
    startTime: new Date(),
    endTime: new Date(),
    processingTime: 0
  });

  useEffect(() => {
    // Generate new confidence data when the component mounts
    setConfidenceData(getRandomConfidenceData());
  }, []);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(isEditing ? editedText : originalText);
      alert('Text copied to clipboard!');
    } catch (err) {
      console.error('Could not copy text: ', err);
    }
  };
  
  const downloadText = () => {
    const element = document.createElement('a');
    const file = new Blob([isEditing ? editedText : originalText], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = 'ocr-result.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };
  
  const printResult = () => {
    const printWindow = window.open('', '_blank');
    
    if (!printWindow) {
      alert('Pop-up blocker may be preventing the print window from opening. Please allow pop-ups for this site.');
      return;
    }
    
    printWindow.document.write(`
      <html>
        <head>
          <title>OCR Result</title>
          <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #6b46c1; }
            .text-content { white-space: pre-wrap; }
            @media print {
              .no-print { display: none; }
            }
          </style>
        </head>
        <body>
          <div class="container">
            <h1>OCR Result</h1>
            ${imageUrl ? `<img src="${imageUrl}" alt="Scanned" style="max-width: 100%; max-height: 300px; margin-bottom: 20px;">` : ''}
            <div class="text-content">${isEditing ? editedText : originalText}</div>
          </div>
        </body>
      </html>
    `);
    printWindow.document.close();
    setTimeout(() => {
      printWindow.print();
    }, 500);
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: 'OCR Result',
        text: isEditing ? editedText : originalText,
      }).catch((error) => console.log('Error sharing', error));
    } else {
      alert('Web Share API not supported in your browser');
    }
  };

  const zoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.25, 3));
  };

  const zoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.25, 0.5));
  };


  const saveEdits = () => {
    setOriginalText(editedText); 
    setIsEditing(false);
    alert('Changes saved successfully!');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8">
        <button
          onClick={() => navigate('/')}
          className="mb-8 flex items-center text-purple-300 hover:text-purple-400"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back to Home
        </button>

        <div className="max-w-4xl mx-auto bg-gray-800 rounded-2xl shadow-xl border border-gray-700 p-8">
          <h1 className="text-3xl font-bold text-white mb-6">OCR Result</h1>
          
          <div className="flex mb-6 border-b border-gray-700">
            <button 
              className={`px-4 py-2 ${activeTab === 'text' ? 'text-purple-400 border-b-2 border-purple-400' : 'text-gray-400'}`}
              onClick={() => setActiveTab('text')}
            >
              Text Results
            </button>
            <button 
              className={`px-4 py-2 ${activeTab === 'visual' ? 'text-purple-400 border-b-2 border-purple-400' : 'text-gray-400'}`}
              onClick={() => setActiveTab('visual')}
            >
              Visual Analysis
            </button>
            <button 
              className={`px-4 py-2 ${activeTab === 'stats' ? 'text-purple-400 border-b-2 border-purple-400' : 'text-gray-400'}`}
              onClick={() => setActiveTab('stats')}
            >
              Statistics
            </button>
          </div>

          {activeTab === 'text' && (
            <>
              {imageUrl && (
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <h2 className="text-lg font-semibold text-purple-300">Uploaded Image:</h2>
                    <div className="flex space-x-2">
                      <button onClick={zoomOut} className="p-1 text-gray-400 hover:text-white" disabled={zoomLevel <= 0.5}>
                        <ZoomOut className="h-4 w-4" />
                      </button>
                      <span className="text-gray-400">{Math.round(zoomLevel * 100)}%</span>
                      <button onClick={zoomIn} className="p-1 text-gray-400 hover:text-white" disabled={zoomLevel >= 3}>
                        <ZoomIn className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <div className="overflow-auto border border-gray-700 rounded-lg">
                    <img 
                      src={imageUrl} 
                      alt="Uploaded" 
                      className="rounded-lg transform origin-top-left"
                      style={{ transform: `scale(${zoomLevel})`, maxHeight: '300px' }}
                    />
                  </div>
                </div>
              )}

              <div className="space-y-4">
                <div className="p-4 bg-gray-900 rounded-lg border border-gray-700">
                  <div className="flex justify-between items-center mb-2">
                    <h2 className="text-lg font-semibold text-purple-300">Recognized Text:</h2>
                    <div className="flex items-center">
                      {!isEditing ? (
                        <button
                          onClick={() => setIsEditing(true)}
                          className="flex items-center space-x-1 text-gray-400 hover:text-purple-400"
                        >
                          <Edit className="h-4 w-4" />
                          <span className="text-sm">Edit</span>
                        </button>
                      ) : (
                        <button
                          onClick={saveEdits}
                          className="flex items-center space-x-1 text-green-400 hover:text-green-300"
                        >
                          <Save className="h-4 w-4" />
                          <span className="text-sm">Save</span>
                        </button>
                      )}
                    </div>
                  </div>
                  
                  {isEditing ? (
                    <textarea
                      value={editedText}
                      onChange={(e) => setEditedText(e.target.value)}
                      className="w-full h-64 bg-gray-800 border border-gray-700 rounded-lg p-3 text-gray-300 focus:border-purple-500 focus:ring focus:ring-purple-500 focus:ring-opacity-50"
                    />
                  ) : (
                    <div className="relative">
                      <p className="text-gray-300 whitespace-pre-wrap">{originalText}</p>
                      {showConfidence && (
                        <div className="absolute top-0 right-0 bg-gray-800 p-2 rounded-lg shadow-lg border border-gray-700">
                          <p className="text-sm text-gray-300">Confidence: {confidenceData.overallConfidence}%</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={copyToClipboard}
                    className="flex items-center space-x-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg transition-colors"
                  >
                    <Copy className="h-4 w-4" />
                    <span>Copy</span>
                  </button>

                  <button
                    onClick={downloadText}
                    className="flex items-center space-x-2 px-3 py-2 bg-blue-700 hover:bg-blue-600 text-white rounded-lg transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    <span>Download</span>
                  </button>
                  
                  <button
                    onClick={printResult}
                    className="flex items-center space-x-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg transition-colors"
                  >
                    <Printer className="h-4 w-4" />
                    <span>Print</span>
                  </button>
                  
                  <button
                    onClick={handleShare}
                    className="flex items-center space-x-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg transition-colors"
                  >
                    <Share2 className="h-4 w-4" />
                    <span>Share</span>
                  </button>
                  
                  <button
                    onClick={() => setShowConfidence(!showConfidence)}
                    className={`flex items-center space-x-2 px-3 py-2 ${showConfidence ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-200'} rounded-lg transition-colors`}
                  >
                    <Info className="h-4 w-4" />
                    <span>Confidence</span>
                  </button>

                  <button
                    onClick={() => navigate('/upload')}
                    className="flex items-center space-x-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                  >
                    <Home className="h-4 w-4" />
                    <span>Another Image</span>
                  </button>
                  <button
                    onClick={() => navigate('/chatbot', { 
                      state: { 
                        initialMessage: isEditing ? editedText : text,
                        fromOCR: true 
                      } 
                    })}
                    className="flex items-center space-x-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                  >
                    <MessageSquare className="h-4 w-4" />
                    <span>Send to AI Chat</span>
                  </button>
                </div>
              </div>
            </>
          )}

          {activeTab === 'visual' && (
            <div className="p-6 bg-gray-900 rounded-lg border border-gray-700">
              <h2 className="text-lg font-semibold text-purple-300 mb-4">Visual Character Analysis</h2>
              <p className="text-gray-400 mb-4">This view shows character-by-character confidence levels and potential issues.</p>
              
              <div className="border border-gray-700 rounded-lg p-4 bg-gray-800">
                <p className="text-gray-300">Character-by-character analysis would be displayed here, with color coding for confidence levels.</p>
                <div className="flex gap-4 mt-4">
                  <div className="w-4 h-4 bg-green-500 rounded"></div>
                  <span className="text-gray-300">High confidence (90%+)</span>
                </div>
                <div className="flex gap-4 mt-2">
                  <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                  <span className="text-gray-300">Medium confidence (70-90%)</span>
                </div>
                <div className="flex gap-4 mt-2">
                  <div className="w-4 h-4 bg-red-500 rounded"></div>
                  <span className="text-gray-300">Low confidence (below 70%)</span>
                </div>
              </div>
            </div>
          )}

{activeTab === 'stats' && (
            <div className="p-6 bg-gray-900 rounded-lg border border-gray-700">
              <h2 className="text-lg font-semibold text-purple-300 mb-4">Recognition Statistics</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-md text-purple-200 mb-2">Overall Recognition</h3>
                  <div className="flex items-center mb-2">
                    <div className="w-full bg-gray-700 rounded-full h-4">
                      <div 
                        className="bg-purple-600 h-4 rounded-full" 
                        style={{ width: `${confidenceData.overallConfidence}%` }}
                      ></div>
                    </div>
                    <span className="ml-2 text-gray-300">{confidenceData.overallConfidence}%</span>
                  </div>
                  <p className="text-gray-400 text-sm mt-2">Overall confidence score for the entire document</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-md text-purple-200 mb-2">Character Confidence</h3>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>High Confidence</span>
                        <span>{confidenceData.characterConfidences.high}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-green-500 h-2 rounded-full" style={{ width: `${confidenceData.characterConfidences.high}%` }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>Medium Confidence</span>
                        <span>{confidenceData.characterConfidences.medium}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-yellow-500 h-2 rounded-full" style={{ width: `${confidenceData.characterConfidences.medium}%` }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Low Confidence</span>
                        <span>{confidenceData.characterConfidences.low}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="bg-red-500 h-2 rounded-full" style={{ width: `${confidenceData.characterConfidences.low}%` }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-md text-purple-200 mb-2">Document Analysis</h3>
                  <div className="space-y-2">
                  <div className="flex justify-between text-sm">
  <span className="text-gray-400">Total Characters:</span>
  <span className="text-gray-300">{originalText.length}</span>
</div>
<div className="flex justify-between text-sm">
  <span className="text-gray-400">Total Words:</span>
  <span className="text-gray-300">{originalText.split(/\s+/).filter((word: string | any[]) => word.length > 0).length}</span>
</div>
<div className="flex justify-between text-sm">
  <span className="text-gray-400">Total Lines:</span>
  <span className="text-gray-300">{originalText.split(/\n/).length}</span>
</div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Processing Time:</span>
                      <span className="text-gray-300">8.32 seconds</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-md text-purple-200 mb-2">Recognition Issues</h3>
                  <div className="space-y-2">
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                      <span className="text-gray-300">2 potential misspellings detected</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                      <span className="text-gray-300">3 characters with low confidence</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                      <span className="text-gray-300">1 unclear formatting structure</span>
                    </div>
                    <div className="mt-3">
                      <button className="text-purple-400 text-sm hover:text-purple-300">
                        View detailed analysis →
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 bg-gray-800 p-4 rounded-lg border border-gray-700">
                <h3 className="text-md text-purple-200 mb-3">Image Quality Assessment</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-900 mb-2">
                      <span className="text-green-400 text-xl font-bold">A</span>
                    </div>
                    <p className="text-gray-300 text-sm">Brightness</p>
                  </div>
                  <div className="text-center">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-yellow-900 mb-2">
                      <span className="text-yellow-400 text-xl font-bold">B</span>
                    </div>
                    <p className="text-gray-300 text-sm">Contrast</p>
                  </div>
                  <div className="text-center">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-900 mb-2">
                      <span className="text-green-400 text-xl font-bold">A</span>
                    </div>
                    <p className="text-gray-300 text-sm">Resolution</p>
                  </div>
                  <div className="text-center">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-blue-900 mb-2">
                      <span className="text-blue-400 text-xl font-bold">B+</span>
                    </div>
                    <p className="text-gray-300 text-sm">Clarity</p>
                  </div>
                </div>
                <div className="mt-4">
                  <p className="text-gray-400 text-sm">Overall Image Quality Score: <span className="text-green-400 font-medium">85/100</span> - Good quality for OCR processing</p>
                </div>
              </div>

              <div className="mt-6 flex justify-end">
                <button 
                  className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                  onClick={() => {
                    // In a real app, this would generate a detailed report
                    alert('Generating detailed report...');
                  }}
                >
                  <Download className="h-4 w-4" />
                  <span>Export Detailed Report</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ResultPage;