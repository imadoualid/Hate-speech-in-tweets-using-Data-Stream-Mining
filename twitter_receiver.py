import tweepy
from tweepy.auth import OAuthHandler
from tweepy import Stream
import socket
import json
import sys

consumer_key    = '2nHji7ad66TYKAvE7RUYYHsmL'
consumer_secret = 'uYTdrwx9OBkm57bVuVRq3AeGojs9yc32PtsAhWjIBq7nhoDNLD'
access_token    = '1161003528489488384-OwPuy3mLdx1wLiYQzL9bC279r6Zzpi'
access_secret   = 'GLXTeKgxDCWoquEg4POO9gl5M0AtvBgWwkoUrdmjvXHey'

class TweetsListener(tweepy.StreamListener):

    def __init__(self, client_socket):
        self.client_socket = client_socket

    def on_data(self, data):
        try:
            message = json.loads( data )
            encoded_message = message['text'].encode('utf-8')
            if(encoded_message):
                print( message['text'].encode('utf-8') )
                self.client_socket.send( message['text'].encode('utf-8') )
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            sys.stderr.write("Rate Limited : There's too many requests for the App")
        else:
            sys.stderr.write("Error : %s " % str(status))
        return True

if __name__ == "__main__":
    skt = socket.socket()
    address = ("127.0.0.1",5555)
    skt.bind(address)

    print("Listening on port: %s" % str(address[1]))

    skt.listen(5)
    client_socket, client_address = skt.accept()

    print("Received request from: " + str(client_address))

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListener(client_socket))
    twitter_stream.filter(track=["football"],
                          languages=["en"])
