import os
import assemblyai as aai                                                                                                      
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions,                                                                                  
    StreamingEvents, StreamingParameters, TurnEvent,
)                                                                                                                             
                                                                                                                            
def on_turn(client, event: TurnEvent):                                                                                        
    print(event.transcript)                                                                                                   
                            
client = StreamingClient(StreamingClientOptions(api_key=os.environ["ASSEMBLYAI_API_KEY"]))                                    
client.on(StreamingEvents.Turn, on_turn)                                                  
client.connect(StreamingParameters(sample_rate=16000, format_turns=True))                                                     
# Use a custom bytes-iterator here, not aai.extras.MicrophoneStream    