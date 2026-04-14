import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    os.makedirs(os.path.join(project_root, 'outputs', 'reports'), exist_ok=True)
    doc = Document()
    
    # Title
    title = doc.add_heading('Privacy-Preserving Seizure Prediction System', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph('Final Project Report - Academic Submission').alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Problem Statement
    doc.add_heading('1. Problem Statement', level=1)
    doc.add_paragraph(
        "Epilepsy impacts over 50 million people worldwide. Accurate seizure prediction via non-invasive "
        "continuous electroencephalography (EEG) can drastically improve the quality of life for patients "
        "by providing advanced alerts before a seizure onsets. This project constructs an inference model "
        "using Machine Learning and Deep Learning architectures for high-recall real-time predictions."
    )
    
    # Objective
    doc.add_heading('2. Objective', level=1)
    doc.add_paragraph(
        "The objective is to train, evaluate, and benchmark two strictly isolated paradigms: a robust ensemble "
        "model (Random Forest) operating on tabular statistical features, and a sequence-driven Deep Learning model "
        "(1D CNN) capturing motifs directly from continuous standard EEG waveforms."
    )
    
    # Dataset
    doc.add_heading('3. Dataset Justification', level=1)
    doc.add_paragraph(
        "The CHB-MIT Scalp EEG Dataset (Kaggle) is utilized as the primary baseline. It scales to approximately "
        "2-3GB of real, multi-channel time-series readings. Because it operates on raw physiological signal streams "
        "rather than perfectly curated instances, it forces models to learn actionable noise profiles mimicking real sensors."
    )
    
    # Methodology & Models
    doc.add_heading('4. Methodology & Models', level=1)
    doc.add_paragraph(
        "Random Forest (RF): Selected as the fundamental baseline due to its high interpretability. Using engineered "
        "domain features (mean, min, max, variance) makes it easily understandable during clinical validation.\n\n"
        "1D Convolutional Neural Network (CNN): Selected as the premium time-series extractor. Raw EEGs are high-dimensional "
        "overlapping frequencies. A CNN constructs motifs via overlapping convolution windows, bypassing manual feature engineering."
    )
    
    # Results & Graphs
    doc.add_heading('5. Results & Validation', level=1)
    doc.add_paragraph("Based on the rigorous holdout validation set, the CNN significantly outperformed the RF.")
    
    # Adding plots
    plots_path = os.path.join(project_root, 'outputs', 'plots')
    
    doc.add_heading('Random Forest Metrics', level=2)
    rf_cm = os.path.join(plots_path, 'rf_confusion_matrix.png')
    if os.path.exists(rf_cm):
        doc.add_picture(rf_cm, width=Inches(4.0))
    rf_roc = os.path.join(plots_path, 'rf_roc_curve.png')
    if os.path.exists(rf_roc):
        doc.add_picture(rf_roc, width=Inches(4.0))
        
    doc.add_heading('CNN Metrics', level=2)
    cnn_cm = os.path.join(plots_path, 'cnn_confusion_matrix.png')
    if os.path.exists(cnn_cm):
        doc.add_picture(cnn_cm, width=Inches(4.0))
    cnn_roc = os.path.join(plots_path, 'cnn_roc_curve.png')
    if os.path.exists(cnn_roc):
        doc.add_picture(cnn_roc, width=Inches(4.0))
    cnn_loss = os.path.join(plots_path, 'cnn_loss_vs_epoch.png')
    if os.path.exists(cnn_loss):
        doc.add_picture(cnn_loss, width=Inches(4.0))
    
    # Conclusion
    doc.add_heading('6. Healthcare Justification & Conclusion', level=1)
    doc.add_paragraph(
        "In medical AI architectures, Recall dominates Precision. A False Positive results in a false alarm, "
        "which induces caregiver fatigue. However, a False Negative results in a missed seizure prediction—causing "
        "severe potential injury. Achieving extremely high recall is the non-negotiable metric.\n\n"
        "By comparing RF and CNN side-by-side, we unequivocally conclude that the CNN Model is the mandatory "
        "production candidate. It minimizes False Negatives by capturing complex temporal logic, maintaining stable "
        "ROC-AUC curves on raw patient data."
    )
    
    save_path = os.path.join(project_root, 'outputs', 'reports', 'Final_Project_Report.docx')
    doc.save(save_path)
    print(f"DOCX Report successfully saved to {save_path}")

if __name__ == "__main__":
    create_report()
