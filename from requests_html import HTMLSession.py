import requests as req
import pandas as pd
from bs4 import BeautifulSoup

def aspx_to_csv(url, output_file):
    response = req.get(url)

    # Check for successful response
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from URL: {url} (Status Code: {response.status_code})")

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table
    tables = soup.find_all('table')
    if not tables:
        raise Exception("No tables found on the ASPX page.")
    table = tables[0]

    # Extract headers
    headers = [header.get_text(strip=True) for header in table.find_all('th')]

    # Extract rows
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        rows.append(cells)

    # Ensure headers and rows have consistent lengths
    for row in rows:
        while len(row) < len(headers):  # Fill missing columns
            row.append('N/A')
    headers = headers[:len(rows[0])]  # Trim headers if needed

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data successfully written to {output_file}")

# Example usage
url = "https://merolagani.com/LatestMarket.aspx"
output_file = "LatestMarketData.csv"
aspx_to_csv(url, output_file)
data = pd.read_csv('LatestMarketData.csv')

# Display the first few rows of the data
print(data)