from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import network.routing

application = ProtocolTypeRouter({
	'websocket': AuthMiddlewareStack(
		URLRouter(
			network.routing.websocket_urlpatterns
		)
	)
})