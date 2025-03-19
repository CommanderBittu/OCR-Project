import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, ImagePlus } from 'lucide-react';
import axios from 'axios';
import ProcessingLoader from '../components/ProcessingLoader';

function UploadPage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('Upload response:', response.data);

      navigate('/result', {
        state: {
          text: response.data.text,
          imageUrl: previewUrl,
          filename: response.data.filename,
          processedImageUrl: response.data.annotated_image
        }
      });
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Failed to recognize text.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {isProcessing && <ProcessingLoader />}
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-2">Upload Image</h1>
          <p className="text-lg text-purple-300">Upload your image containing handwritten text</p>
        </div>
        <div className="max-w-3xl mx-auto bg-gray-800 rounded-2xl shadow-xl border border-gray-700 p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-purple-400 transition-colors">
              <input
                type="file"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
                accept="image/*"
              />
              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center space-y-4">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="max-h-64 rounded-lg" />
                ) : (
                  <>
                    <ImagePlus className="h-16 w-16 text-purple-400" />
                    <span className="text-purple-300">Click or drag to upload an image</span>
                  </>
                )}
              </label>
            </div>
            <button
              type="submit"
              disabled={!selectedFile || isProcessing}
              className={`w-full py-3 px-6 rounded-lg flex items-center justify-center space-x-2 text-white font-medium
                ${!selectedFile || isProcessing ? 'bg-gray-600 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700'}`}
            >
              <Upload className="h-5 w-5" />
              <span>{isProcessing ? 'Processing...' : 'Recognize Text'}</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;