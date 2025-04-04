#!/usr/bin/env python3
"""
Convert Rmd files to HTML with basic styling for preview purposes.
This is a temporary solution until R can be installed.
"""

import os
import re
import sys
from pathlib import Path

def convert_rmd_to_html(rmd_file, output_file=None):
    """Convert an Rmd file to HTML with styling for preview."""
    if output_file is None:
        output_file = rmd_file.replace('.Rmd', '_preview.html')
    
    with open(rmd_file, 'r') as f:
        content = f.read()
    
    # Extract YAML header
    yaml_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    yaml_header = yaml_match.group(1) if yaml_match else ""
    
    # Parse title, author, date
    title = re.search(r'title:\s*"([^"]*)"', yaml_header)
    author = re.search(r'author:\s*"([^"]*)"', yaml_header)
    date = re.search(r'date:\s*"([^"]*)"', yaml_header)
    
    title = title.group(1) if title else "Dashboard"
    author = author.group(1) if author else ""
    date = date.group(1) if date else ""
    
    # Read CSS
    css_path = os.path.join(os.path.dirname(rmd_file), 'styles.css')
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            css_content = f.read()
    
    # Replace R code chunks with placeholders for visualization
    content = re.sub(r'```\{r[^}]*\}(.*?)```', r'<div class="visualization-placeholder"><div class="placeholder-text">Interactive Visualization<br>(Will render in R)</div></div>', content, flags=re.DOTALL)
    
    # Basic Markdown to HTML conversion
    # Headers
    content = re.sub(r'^#\s+(.*?)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^##\s+(.*?)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^####\s+(.*?)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
    
    # Bold and Italic
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    
    # Lists
    content = re.sub(r'^-\s+(.*?)$', r'<li>\1</li>', content, flags=re.MULTILINE)
    content = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\g<0></ul>', content, flags=re.DOTALL)
    
    # Replace *** dividers with section breaks
    content = re.sub(r'^\*\*\*$', r'<hr class="section-divider">', content, flags=re.MULTILINE)
    
    # Basic structure for flexdashboard-like layout
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>{title}</title>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
      <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
      <style>
        {css_content}
        
        /* Additional preview styles */
        body {{
          padding-top: 70px;
          background-color: #f5f7fa;
        }}
        
        .container {{
          max-width: 1200px;
          background-color: white;
          padding: 30px;
          border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
          margin-bottom: 50px;
        }}
        
        .header {{
          text-align: center;
          margin-bottom: 30px;
        }}
        
        .visualization-placeholder {{
          background-color: #f8f9fa;
          border: 1px dashed #adb5bd;
          border-radius: 6px;
          height: 300px;
          margin: 20px 0;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
        }}
        
        .placeholder-text {{
          color: #6c757d;
          font-size: 1.2rem;
        }}
        
        .section-divider {{
          margin: 40px 0;
          border-top: 1px solid #e9ecef;
        }}
        
        h1, h2, h3, h4 {{
          color: #2c3e50;
          margin-top: 1.5em;
        }}
        
        h1 {{
          font-size: 2.2rem;
          border-bottom: 2px solid #3498db;
          padding-bottom: 10px;
        }}
        
        h2 {{
          font-size: 1.8rem;
        }}
        
        h3 {{
          font-size: 1.4rem;
        }}
        
        ul {{
          margin-bottom: 20px;
        }}
        
        .navbar {{
          background-color: #1a4a72;
        }}
        
        .navbar-brand {{
          color: white;
          font-weight: bold;
        }}
        
        .footer {{
          text-align: center;
          margin-top: 50px;
          padding-top: 20px;
          border-top: 1px solid #e9ecef;
          color: #6c757d;
        }}
      </style>
    </head>
    <body>
      <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container-fluid">
          <span class="navbar-brand">{title}</span>
        </div>
      </nav>
      
      <div class="container">
        <div class="header">
          <h1>{title}</h1>
          <p><strong>{author}</strong> • {date}</p>
        </div>
        
        <div class="dashboard-content">
          {content}
        </div>
        
        <div class="footer">
          <p>This is a preview of the flexdashboard. For full interactive features, render with R.</p>
          <p>GNN-CD Framework • {date}</p>
        </div>
      </div>
      
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Created preview HTML: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        rmd_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_rmd_to_html(rmd_file, output_file)
    else:
        print("Usage: python convert_rmd.py path/to/file.Rmd [output.html]")