# =========================================================
# PDF REPORT GENERATOR
# RESPONSIBLE AI PROJECT
# =========================================================

import os

from reportlab.platypus import (

    SimpleDocTemplate,

    Paragraph,

    Spacer,

    Table,

    TableStyle
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.lib.pagesizes import letter

from reportlab.lib import colors


# =========================================================
# GENERATE PDF REPORT
# =========================================================

def generate_pdf_report(

    output_path,

    content,

    bias_table_data=None
):

    print(
        "\n========== PDF REPORT GENERATION =========="
    )

    try:

        # =================================================
        # CREATE OUTPUT DIRECTORY
        # =================================================

        os.makedirs(

            os.path.dirname(output_path),

            exist_ok=True
        )

        # =================================================
        # CREATE PDF DOCUMENT
        # =================================================

        doc = SimpleDocTemplate(

            output_path,

            pagesize=letter
        )

        # =================================================
        # STYLES
        # =================================================

        styles = getSampleStyleSheet()

        elements = []

        # =================================================
        # TITLE
        # =================================================

        title = Paragraph(

            "<b>Responsible AI Report</b>",

            styles["Title"]
        )

        elements.append(title)

        elements.append(
            Spacer(1, 20)
        )

        # =================================================
        # CONTENT
        # =================================================

        lines = content.split("\n")

        for line in lines:

            if line.strip() == "":

                elements.append(
                    Spacer(1, 10)
                )

                continue

            paragraph = Paragraph(

                line,

                styles["BodyText"]
            )

            elements.append(
                paragraph
            )

        # =================================================
        # BIAS TABLE
        # =================================================

        if bias_table_data is not None:

            elements.append(
                Spacer(1, 20)
            )

            elements.append(

                Paragraph(

                    "<b>Bias Analysis Table</b>",

                    styles["Heading2"]
                )
            )

            elements.append(
                Spacer(1, 10)
            )

            table = Table(

                bias_table_data,

                colWidths=[
                    140,
                    70,
                    70,
                    90
                ]
            )

            table.setStyle(

                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),

                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.black
                    ),

                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),

                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.black
                    ),

                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.whitesmoke
                    )
                ])
            )

            elements.append(
                table
            )

        # =================================================
        # BUILD PDF
        # =================================================

        doc.build(
            elements
        )

        # =================================================
        # SUCCESS MESSAGE
        # =================================================

        print(

            f"\nPDF Report Saved:\n{output_path}"
        )

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print(
            "\nPDF Report Generation Failed"
        )

        print(e)