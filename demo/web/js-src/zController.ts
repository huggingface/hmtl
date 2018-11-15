
const ENDPOINT = 
	// "https://coref.huggingface.co/coref"
	'https://hugging-nlp.huggingface.co/jmd'
;
const DEFAULT_NLP_TEXT = () => {
	const items = [
		// `I love my father and my mother. They work hard. She is always nice but he is sometimes rude.`,
		// `My sister is swimming with her classmates. They are not bad, but she is better. I love watching her swim.`,
		// `My mother's name is Sasha, she likes dogs.`,
		// `My name is Jean-Claude, I love mushrooms.`,
		// `My mother's name is Sasha, she likes dogs and works at Google.`,
		`In Boston, Michelle used to run with John Lennon. He was as slow as a snail, but she was as fast as a train, probably because she worked at a running shop.`,
	];
	return items[Math.floor(Math.random()*items.length)];
}

const loading = () => {
	document.body.classList.toggle('loading');
};

const toggleDebug = () => {
	document.body.classList.toggle('debug');
	const icons = document.querySelectorAll('.svg-checkbox');
	(<any>icons).forEach((icon) => {
		icon.classList.toggle('hide');
	});
	/// local storage
	window.localStorage.setItem('debug', document.body.classList.contains('debug').toString());
};

const nlp = new HuggingNlp(ENDPOINT, {
	onStart: loading,
	onSuccess: loading,
});

const getQueryVar = (key: string) => {
	const query = window.location.search.substring(1);
	const params = query.split('&').map(param => param.split('='));
	for (const param of params) {
		if (param[0] === key) { return decodeURIComponent(param[1]); }
	}
	return undefined;
}

const updateURL = (text) => {
	history.pushState({ text: text }, "", `?text=${encodeURIComponent(text)}`);
}

const updateLinks = (text: string) => {
	const corenlpDiv = document.querySelector('.js-corenlp');
	if (corenlpDiv) {
		corenlpDiv.setAttribute(
			'href',
			`http://corenlp.run/#text=${encodeURIComponent(text)}`
		);
	}
	const displacyDiv = document.querySelector('.js-displacy');
	if (displacyDiv) {
		displacyDiv.setAttribute(
			'href',
			`https://explosion.ai/demos/displacy-ent?text=${encodeURIComponent(text)}`
		);
	}
};

document.addEventListener('DOMContentLoaded', () => {
	const $input        = document.querySelector('input.input-message') as HTMLInputElement;
	const $form         = document.querySelector('form.js-form') as HTMLFormElement;
	const $checkbox     = document.querySelector('.js-checkbox') as HTMLElement;
	
	{
		// Initial text
		const queryText = getQueryVar('text');
		if (queryText) {
			$input.value = queryText;
		}
		const text = queryText || DEFAULT_NLP_TEXT();
		nlp.parse(text);
		// nlp.dummyParse();
		updateLinks(text);
	}
	
	$input.addEventListener('keydown', (evt) => {
		if (evt.charCode === 13) {
			// 13 is the Enter key
			evt.preventDefault();
			$form.submit();
		}
	});
	
	$form.addEventListener('submit', (evt) => {
		evt.preventDefault();
		const text = ($input.value.length > 0)
			? $input.value
			: DEFAULT_NLP_TEXT();
		updateURL(text);
		nlp.parse(text);
		updateLinks(text);
	});
	
	// $checkbox.addEventListener('click', () => {
	// 	toggleDebug();
	// });
	
	// Highlight
	const __handleHashChange = () => {
		if (window.location.hash === "") {
			return ;
		}
		const hash = window.location.hash.slice(1);
		const d = document.querySelector(`.task.${hash}`);
		const wasHighlighted = (d && d.classList.contains('highlighted'));
		/// This is not working as intended.
		for (const div of Array.from(document.querySelectorAll('.task.highlighted'))) {
			div.classList.remove('highlighted');
		}
		if (d && !wasHighlighted) {
			d.classList.add('highlighted');
		}
	}
	window.onhashchange = () => {
		__handleHashChange();
	};
	__handleHashChange();
	
	// Turn on debug mode by default, unless the string `false` is stored in local storage:
	if (window.localStorage.getItem('debug') !== 'false') {
		toggleDebug();
	}
});


