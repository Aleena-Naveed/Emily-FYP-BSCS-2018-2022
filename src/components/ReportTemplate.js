
import { Button } from '@material-ui/core';
import { PDFExport, savePDF } from '@progress/kendo-react-pdf';
import { useRef } from 'react';

function ExportExample() {
    const pdfExportComponent = useRef(null);
    const contentArea = useRef(null);

    const handleExportWithComponent = (event) => {
        pdfExportComponent.current.save();
    }

    const handleExportWithFunction = (event) => {
        savePDF(contentArea.current, { paperSize: "A4" });
    }

    return (
        <div className="app-content">
            <PDFExport ref={pdfExportComponent} paperSize="A4">
                <div >
                    <h1>KendoReact PDF Processing</h1>
                    <p>This is an example of text that may be <span className="neat-style">styled</span></p>
                    <div className="button-area">
                        <Button primary={true} onClick={handleExportWithComponent}>Export with Component</Button>
                        <Button onClick={handleExportWithFunction}>Export with Method</Button>
                    </div>
                </div>
            </PDFExport>
        </div>
    );
}

export default ExportExample;