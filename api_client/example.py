"""Example usage of the API client."""

import asyncio
from api_client import APIClient, APIConfig


def sync_example():
    """Example of synchronous API client usage."""
    # Create configuration
    config = APIConfig(
        base_url="https://api.example.com",
        connection_timeout=10.0,
        read_timeout=30.0,
        max_retries=2,
    )
    
    # Create client with configuration
    with APIClient(config) as client:
        try:
            # GET request
            response = client.get("/users")
            print(f"GET response: {response}")
            
            # POST request
            data = {"name": "John Doe", "email": "john@example.com"}
            response = client.post("/users", data=data)
            print(f"POST response: {response}")
            
            # PUT request
            data = {"name": "Jane Doe", "email": "jane@example.com"}
            response = client.put("/users/1", data=data)
            print(f"PUT response: {response}")
            
            # DELETE request
            response = client.delete("/users/1")
            print(f"DELETE response: {response}")
            
        except Exception as e:
            print(f"Error: {e}")


async def async_example():
    """Example of asynchronous API client usage."""
    # Create configuration with custom headers
    config = APIConfig(
        base_url="https://api.example.com",
        connection_timeout=10.0,
        read_timeout=30.0,
        max_retries=2,
    ).with_headers({
        "Authorization": "Bearer your-token-here",
        "X-Custom-Header": "custom-value"
    })
    
    # Create client with configuration
    async with APIClient(config) as client:
        try:
            # Async GET request
            response = await client.aget("/users")
            print(f"Async GET response: {response}")
            
            # Async POST request
            data = {"name": "John Doe", "email": "john@example.com"}
            response = await client.apost("/users", data=data)
            print(f"Async POST response: {response}")
            
            # Async PUT request
            data = {"name": "Jane Doe", "email": "jane@example.com"}
            response = await client.aput("/users/1", data=data)
            print(f"Async PUT response: {response}")
            
            # Async DELETE request
            response = await client.adelete("/users/1")
            print(f"Async DELETE response: {response}")
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run examples."""
    print("=== Synchronous Example ===")
    sync_example()
    
    print("\n=== Asynchronous Example ===")
    asyncio.run(async_example())


if __name__ == "__main__":
    main()