import requests
from bs4 import BeautifulSoup

def get_real_data(company_code):
    # URL of the historical prices page
    url = f"https://www.wsj.com/market-data/quotes/BD/XDHA/{company_code}/historical-prices"

    # Send an HTTP request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    print(f"\nSoup: {soup}\n")
    
    # Find the table with id="historical_data_table"
    historical_table = soup.find("div", id="historical_data_table")
    # historical_table = soup.find("table", class_="cr_dataTable")

    print(f"\n\n{historical_table}\n\n")

    # Find all rows in the table body
    rows = historical_table.tbody.find_all("tr")

    # Extract data from each row
    for row in rows:
        columns = row.find_all("td")
        date = columns[0].text.strip()
        closing_price = columns[4].text.strip()
        print(f"Date: {date}, Closing Price: {closing_price}")

    # Note: You may need to adjust the column indices based on the actual structure of the table
