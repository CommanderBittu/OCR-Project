import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, ArrowRight, CheckCircle2, Upload, FileText, Sparkles } from 'lucide-react';

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex justify-center mb-8">
            <Eye className="h-20 w-20 text-purple-400" />
          </div>
          <h1 className="text-6xl font-bold text-white mb-6">Drishti</h1>
          <p className="text-xl text-purple-300 mb-8">
            Transform your handwritten text into digital format instantly with our advanced OCR technology
          </p>
          <p className="text-gray-300 mb-12 max-w-2xl mx-auto">
            Our powerful OCR engine accurately recognizes handwritten text from images, making it easy to digitize your notes, documents, and more.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="inline-flex items-center px-8 py-4 text-lg font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors duration-300"
          >
            Get Started
            <ArrowRight className="ml-2 h-5 w-5" />
          </button>
        </div>

        {/* How It Works Section */}
        <div className="max-w-5xl mx-auto mt-32">
          <h2 className="text-3xl font-bold text-white text-center mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="relative">
              <div className="bg-gray-800 p-8 rounded-xl shadow-xl border border-gray-700 text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-purple-600/20 mb-6">
                  <Upload className="h-8 w-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-semibold text-purple-300 mb-3">1. Upload Image</h3>
                <p className="text-gray-300">Take a photo or upload an image of your handwritten text</p>
              </div>
              <div className="hidden md:block absolute top-1/2 left-full w-16 border-t-2 border-dashed border-purple-400/30 -translate-y-1/2"></div>
            </div>
            <div className="relative">
              <div className="bg-gray-800 p-8 rounded-xl shadow-xl border border-gray-700 text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-purple-600/20 mb-6">
                  <Sparkles className="h-8 w-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-semibold text-purple-300 mb-3">2. Process</h3>
                <p className="text-gray-300">Our AI analyzes and converts your handwriting to digital text</p>
              </div>
              <div className="hidden md:block absolute top-1/2 left-full w-16 border-t-2 border-dashed border-purple-400/30 -translate-y-1/2"></div>
            </div>
            <div className="bg-gray-800 p-8 rounded-xl shadow-xl border border-gray-700 text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-purple-600/20 mb-6">
                <FileText className="h-8 w-8 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-purple-300 mb-3">3. Get Results</h3>
              <p className="text-gray-300">Copy and use your digitized text anywhere you need</p>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-8 mt-32">
          <div className="bg-gray-800 p-6 rounded-xl shadow-xl border border-gray-700">
            <h3 className="text-xl font-semibold text-purple-300 mb-3">Quick & Easy</h3>
            <p className="text-gray-300">Upload your image and get digitized text in seconds</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-xl shadow-xl border border-gray-700">
            <h3 className="text-xl font-semibold text-purple-300 mb-3">Accurate Results</h3>
            <p className="text-gray-300">Advanced OCR technology ensures high accuracy</p>
          </div>
          <div className="bg-gray-800 p-6 rounded-xl shadow-xl border border-gray-700">
            <h3 className="text-xl font-semibold text-purple-300 mb-3">Copy & Share</h3>
            <p className="text-gray-300">Easily copy and share your digitized text</p>
          </div>
        </div>

        {/* Testimonials Section */}
        <div className="max-w-6xl mx-auto mt-32">
          <h2 className="text-3xl font-bold text-white text-center mb-12">What Our Users Say</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gray-800/50 p-8 rounded-xl shadow-xl backdrop-blur">
              <div className="flex items-start gap-4">
                <img
                  src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=100&h=100"
                  alt="User"
                  className="w-12 h-12 rounded-full object-cover"
                />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="text-purple-300 font-semibold">Sarah Chen</h4>
                    <div className="flex gap-1">
                      {[...Array(5)].map((_, i) => (
                        <CheckCircle2 key={i} className="h-4 w-4 text-purple-400" />
                      ))}
                    </div>
                  </div>
                  <p className="text-gray-300 italic">
                    "Drishti has been a game-changer for digitizing my handwritten notes. The accuracy is impressive, and it saves me so much time!"
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-gray-800/50 p-8 rounded-xl shadow-xl backdrop-blur">
              <div className="flex items-start gap-4">
                <img
                  src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=100&h=100"
                  alt="User"
                  className="w-12 h-12 rounded-full object-cover"
                />
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="text-purple-300 font-semibold">Michael Rodriguez</h4>
                    <div className="flex gap-1">
                      {[...Array(5)].map((_, i) => (
                        <CheckCircle2 key={i} className="h-4 w-4 text-purple-400" />
                      ))}
                    </div>
                  </div>
                  <p className="text-gray-300 italic">
                    "The interface is intuitive, and the conversion process is lightning fast. Exactly what I needed for my work!"
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;