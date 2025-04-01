import React, { useState } from "react";
import UploadFile from "./components/UploadFile";
import ResponseAnalysis from "./components/ResponseAnalysis";

const App = () => {
    const [analysisData, setAnalysisData] = useState(null);

    return (
        <div>
            <h1>Art & Emotion NLP Analysis</h1>
            <UploadFile onUpload={setAnalysisData} />
            {analysisData && <ResponseAnalysis results={analysisData.results} topics={analysisData.topics} />}
        </div>
    );
};

export default App;
