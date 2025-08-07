import pyodbc

def get_connection():
    """
    Connect to SQL Server using the working config.
    """
    driver = "SQL Server"  # You can change this to ODBC Driver 17/18 if available
    server = "localhost"
    database = "LioxsaPlatform"

    conn_str = (
        f"Driver={{{driver}}};"
        f"Server={server};"
        f"Database={database};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )

    try:
        conn = pyodbc.connect(conn_str, timeout=5)
        print(f"✅ Connected to SQL Server at {server}")
        return conn
    except Exception as e:
        raise ConnectionError(f"❌ Connection failed: {e}")

def find_file_by_id(file_id: str):
    """
    Query file info by Id from FileData table.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT [Id],
           [Content],
           [FileAddress],
           [MimeType],
           [FileName],
           [FileSize],
           [CreateDateTime],
           [UpdateDateTime],
           [CreateUserId],
           [LastUpdateUserId]
    FROM [Blob].[FileData]
    WHERE [Id] = ?
    """

    cursor.execute(query, (file_id,))
    row = cursor.fetchone()

    if row:
        print("📦 File Found:")
        print(f"ID: {row.Id}")
        print(f"FileName: {row.FileName}")
        print(f"MimeType: {row.MimeType}")
        print(f"FileSize: {row.FileSize}")
        print(f"Created: {row.CreateDateTime}")
        print(f"Updated: {row.UpdateDateTime}")
    else:
        print("❌ No file found with this ID.")

    conn.close()

if __name__ == "__main__":
    # ✅ Replace with the ID you're searching for
    file_id = "6AFA0F08-EFD5-4C7B-888B-B72163CE433D"
    find_file_by_id(file_id)
