function show_message(message,success=0) {
	var log = $("#log");
	if (success==0)
		log.addClass("error");
	else
		log.addClass("success");
	log.html(message);
	log.removeAttr("hidden");

	setTimeout(function(){
		$("#log").removeClass();
		$("#log").attr('hidden','true');
	},5000);
}

function get_only_data(form) {
	data = {};
	for (tag of form.children)
		if (tag.hasAttribute('value'))
			data[tag.name]=tag.value;
}