import React from "react";

const ResponseAnalysis = ({ results, topics }) => {
    return (
        <div>
            <h2>Analysis Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Response</th>
                        <th>Sentiment</th>
                        <th>Emotion</th>
                    </tr>
                </thead>
                <tbody>
                    {results.map((row, index) => (
                        <tr key={index}>
                            <td>{row.Response}</td>
                            <td>{row.Sentiment}</td>
                            <td>{row.Emotion}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <h2>Extracted Topics</h2>
            <ul>
                {topics.map((topic, index) => (
                    <li key={index}>{topic}</li>
                ))}
            </ul>
        </div>
    );
};

export default ResponseAnalysis;
