import requests
import json

ds_ids = {"GEOMEAN_CELL_STATS": 860, "GEOMEAN_BIOCHEM_STATS": 912}


def get_api_query_text():
    auth_url = "https://dotmatics.kinnate.com/browser/api/authenticate/requestToken?expiration=600"
    auth_params = ("testadmin", "Fountadmin1!")

    try:
        auth_response = requests.get(auth_url, auth=auth_params)
        auth_response.raise_for_status()
        token = auth_response.content.decode().strip('"')
    except requests.exceptions.RequestException as e:
        print("Authentication failed. Error:", str(e))
        return None

    sql_stmts = {}
    for ds, id in ds_ids.items():
        api_url = (
            f"https://dotmatics.kinnate.com/browser/api/datasources/{id}?token={token}"
        )

        query_text = ""
        try:
            api_response = requests.get(api_url)
            if api_response.status_code == 200:
                api_data = json.loads(api_response.content)
                if isinstance(api_data, list) and len(api_data) > 0:
                    first_item = api_data[0]
                    query_text = first_item["queryText"]
                else:
                    print("Invalid API response format")
            else:
                print(
                    f"API request failed. Error: {api_response.status_code} {api_response.text}"
                )
        except requests.exceptions.RequestException as e:
            print("API request failed. Error:", str(e))
            raise

        # query_text = query_text.replace("SELECT", "WHERE COMPOUND_ID = 'FT002787'")
        sql_stmts[ds] = query_text

    return sql_stmts


if __name__ == "__main__":
    print(get_api_query_text())
