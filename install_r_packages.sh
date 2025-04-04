#!/bin/bash
# Script to install R and required packages for rendering the UNSW dashboard

# Install R
echo "Installing R..."
sudo apt-get update
sudo apt-get install -y r-base r-base-dev

# Install required R packages
echo "Installing required R packages..."
R -e "install.packages(c('flexdashboard', 'knitr', 'kableExtra', 'ggplot2', 'plotly', 'DT', 'dplyr', 'tidyr', 'viridis', 'hrbrthemes', 'circlize', 'networkD3', 'scales', 'rmarkdown'), repos='https://cloud.r-project.org/')"

# Create script to render dashboard
echo "Creating render script..."
cat > /home/braden/gnn-cd/render_dashboard.R << EOL
#!/usr/bin/env Rscript
library(rmarkdown)
render("/home/braden/gnn-cd/unsw_results.Rmd", output_file = "unsw_dashboard_full.html")
cat("Dashboard rendered as unsw_dashboard_full.html\n")
EOL

chmod +x /home/braden/gnn-cd/render_dashboard.R

echo "Setup complete! To render the dashboard, run:"
echo "  ./render_dashboard.R"
echo ""
echo "For now, you can view the preview version at:"
echo "  /home/braden/gnn-cd/unsw_dashboard.html"