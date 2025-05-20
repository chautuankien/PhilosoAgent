from src.philosoagent.Phisoloagent import graph
import asyncio

async def get_response():
    output = await graph.ainvoke(
        {
            "messages": ["Hello, how are you?"],
        }
    )
    print(output)

asyncio.run(get_response())