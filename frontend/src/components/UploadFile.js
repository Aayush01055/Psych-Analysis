import React, { useState } from "react";

const UploadFile = ({ onUpload }) => {
    const [file, setFile] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://localhost:5000/analyze", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        onUpload(data);
    };

    return (
        <div>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload & Analyze</button>
        </div>
    );
};

export default UploadFile;
