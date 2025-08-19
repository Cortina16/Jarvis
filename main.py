from elevenlabs import play
from google import genai
from dotenv import load_dotenv
from google.genai import types
from elevenlabs.client import ElevenLabs
import speech_recognition as sr
import tools
import os
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
conversationHistory = []

get_time = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_time",
            description="Returns the current date and time to help with greetings and scheduling",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {},
            'required' : [],
            }
        ),
    ]
)

main_controller_spotify = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="main_controller_spotify",
            description="control spotify playback functions. if no device found, use device issuing commands. :param action: what is going to be performed. :param title: name of the spotify playback title. Should not be filled out if simply unpausing  :param form: what playback medium is being listened to    :return: an action.",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'action' : {
                        'type' : 'string',
                        'description' : "what is going to be performed. Available options are as follows: play, used to unpause music, or to request to play other music. get_track_info, in which one gets information about the current track. pause, which pauses current music. skip_to_next_track, which skips the current track. and finally skip_to_previous_track, which skips but backwards to play the last song.",
                    },
                    'title' : {
                        'type' : 'string',
                        'description' : "what the name of the requested track should be when searching. Only used in the event that action is 'play', and new music is requested"
                    },
                    'form' : {
                        'type' : 'string',
                        'description' : "what type of music should be played. should only be filled out in the event that the action 'play' is used. available options for this are: album, for albums. track, for tracks. and playlist, for playlists."

                    },
                    'amount' : {
                        'type' : 'number',
                        'description' : "amount of songs to skip in the event that action is song. only fill this out of skipping songs."
                    }
                },
                'required' : ['action'],
            }
        ),
    ]
)

web_search = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="web_search",
            description="Search the internet for a provided query and return the result",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'query': {
                        'type' : 'string',
                        'description' : 'The query to search for',
                    }
                },
                'required' : ['query'],
            }
        )
    ]
)

get_weather = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Returns the current weather condition",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'location': {
                        'type' : 'string',
                        'description' : 'The location of the desired weather information',
                    }
                },
                'required' : ['location'],
            }
        )
    ]
)

start_timer = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="start_timer",
            description="Start the timer for the desired time",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'duration' : {
                        'type' : 'number',
                        'description' : "how long to start the timer for",
                    }
                },
                'required' : ['duration'],
            }
        )
    ]
)

search_files = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_files",
            description="Search the current computer for any requested files that may exist",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'query': {
                        'type' : 'string',
                        'description' : 'The files/folder to search the computer for',
                    },
                    'result_amount': {
                        'type': 'number',
                        'description': "how many files to show up in the returned results",
                    }
                },
                'required' : ['query'],
            }
        )
    ]
)

run_program = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="run_program",
            description="Search the current computer for any program that may exist and is desired to run",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'query': {
                        'type' : 'string',
                        'description' : 'the program to attempt to run.',
                    },
                },
                'required' : ['query'],
            }
        )
    ]
)

open_tab = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="open_tab",
            description="open a tab to the desired url in a web browser",
            parameters_json_schema={
                'type' : 'object',
                'properties' : {
                    'url': {
                        'type' : 'string',
                        'description' : 'the web url to open a tab to.',
                    },
                    'web_browser': {
                        'type' : 'string',
                        'description' : 'the web browser to use to open the tab. default option is: firefox regular, which is firefox non private browsing. the second option is: firefox incognito, which is firefox private browsing. the third option is chrome, which is normal google chrome. ',
                    },
                },
                'required' : ['url'],
            }
        )
    ]
)




functionMap = {
    "get_time": tools.getTime,
    "web_search": tools.web_search,
    "get_weather": tools.get_weather,
    "main_controller_spotify": tools.main_controller_spotify,
    "start_timer": tools.start_timer,
    "search_files": tools.search_files,
    "run_program": tools.run_program,
    "open_tab" : tools.open_tabs,
}
tools_list = [get_time, web_search, get_weather, main_controller_spotify, start_timer, search_files, run_program, open_tab]
print(list(functionMap.keys()))
def askgemini(question):
    """
    Sends a question to the Gemini api and stores history
    """
    conversationHistory.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    final_response_text = None
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversationHistory,
        config=types.GenerateContentConfig(
            system_instruction="You are an AI assistant named Jarvis like the Jarvis from Iron Man. Address user as 'Sir' when needed. Don't be robotic. Keep your responses short, preferably under 20 words. The date and time after each query is information. Subtly mention it like 'Good evening' or 'You're up late'",
            temperature=0.5,
            tools=tools_list,
        )
    )
    function_handled = False
    function_calls = []
    function_results = []
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_calls.append(part.function_call)
    if function_calls:
        for function_call in function_calls:
            function_to_call = functionMap.get(function_call.name)
            if function_to_call:
                function_args = getattr(function_call, 'args', {})
                function_result = function_to_call(**function_args)
                function_results.append({
                    "name": function_call.name,
                    "call": function_call,
                    "result": function_result
                })
            else:
                function_results.append({
                    "name": function_call.name,
                    "call": function_call,
                    "result": f"Sir I am sorry but I do not appear to have {function_call.name} available at the moment."
                })
        contents_for_second_call = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=question)]
            ),
            types.Content(
                role="model",
                parts=[{"function_call": fc["call"]} for fc in function_results]
            ),
        ]
        for results in function_results:
            contents_for_second_call.append(
                types.Content(
                    role="function",
                    parts=[types.Part.from_function_response(
                        name=results["name"],
                        response={"result": results["result"]}
                    )]
                )
            )
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_for_second_call,
            config=types.GenerateContentConfig(
                system_instruction="You are an AI assistant named Jarvis like the Jarvis from Iron Man. Address user as 'Sir' when needed. Don't be robotic. Keep your responses short, preferably under 20 words. The date and time after each query is information. Subtly mention it like 'Good evening' or 'You're up late'",
                temperature=0.5,
                tools=tools_list,
            )
        )

        final_response_text = final_response.text
    else:
        final_response_text = response.text
    conversationHistory.append(types.Content(role="model", parts=[types.Part.from_text(text=final_response_text)]))
    return final_response_text

    # else:
    #     final_response_text=response.text
    #     conversationHistory.append(types.Content(role="model", parts=[types.Part.from_text(text=final_response_text)]))
    #     return final_response_text

#----------TTS SYSTEM---------#
def tts_func(text: str):
    elevenlabs = ElevenLabs(
      api_key=os.getenv("ELEVEN_LABS_KEY_2"),
    )
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    return audio

def main():
    while True:
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.recognize_google(r.listen(source))
            print(f"You: {audio}")
            output = askgemini(audio)
            print(f"Jarvis: {output}")
            play(tts_func(output))
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Earth API; {0}".format(e))
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()