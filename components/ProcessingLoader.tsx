import React from 'react';
import { Sparkles } from 'lucide-react';

function ProcessingLoader() {
  return (
    <div className="fixed inset-0 bg-gray-900/90 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="text-center">
        <div className="relative">
          {/* Outer ring animation */}
          <div className="absolute inset-0 rounded-full border-4 border-purple-400/30 animate-[ping_2s_infinite]"></div>
          
          {/* Inner spinning circle */}
          <div className="relative w-20 h-20 rounded-full border-4 border-t-purple-400 border-r-purple-400 border-b-purple-400/30 border-l-purple-400/30 animate-spin">
            <Sparkles className="absolute inset-0 m-auto h-8 w-8 text-purple-400 animate-pulse" />
          </div>
        </div>
        
        {/* Loading text */}
        <div className="mt-8 space-y-3">
          <h3 className="text-xl font-semibold text-white">Processing Your Image</h3>
          <p className="text-purple-300">Our Model is analyzing your handwriting...</p>
          
          {/* Loading dots */}
          <div className="flex justify-center gap-1">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-purple-400"
                style={{
                  animation: `bounce 1.4s infinite ease-in-out`,
                  animationDelay: `${i * 0.04}s`
                }}
              ></div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ProcessingLoader;