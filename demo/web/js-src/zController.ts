
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
		`In 2009, ABC increased its margin by 10%. The company used to manufacture its car in Thailand but moved the factories to China.`,
		`Mark is from Seattle. But housing is so expensive in San Francisco that he used to sleep in the garage of a house.`,
		`Robert was stuck at the airport because of the snow storm. He missed the wedding of his daughter.`,
		`In Boston, Michelle used to run with John Lennon. He was as slow as a snail, but she was as fast as a train, probably because she worked at a running shop.`,
	];
	return items[Math.floor(Math.random()*items.length)];
}

const toggleLoading = (on: boolean) => {
	if (on) {
		document.body.classList.add('loading');
	} else {
		document.body.classList.remove('loading');
	}
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
	onStart:   () => toggleLoading(true),
	onSuccess: () => toggleLoading(false),
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
	
	let pendingText: string | undefined;
	const __launchParseRequest = (text: string) => {
		if (text === pendingText) {
			return ;
		}
		pendingText = text;
		nlp.abortAllPending();
		nlp.parse(text);
		// nlp.dummyParse();
		updateLinks(text);
	};
	
	
	{
		// Initial text
		const queryText = getQueryVar('text');
		if (queryText) {
			$input.value = queryText;
		}
		const text = queryText || DEFAULT_NLP_TEXT();
		__launchParseRequest(text);
	}
	
	
	const __checkInstantSearch = () => {
		const text = $input.value;
		if (text === "") {
			return ;
		}
		if (/[^\w]$/.test(text)) {
			console.log(`lauching instant search with "${text}"`);
			__launchParseRequest(text);
		} else {
			setTimeout(() => {
				if ($input.value === text) {
					console.log(`[debounced] lauching instant search with "${text}"`);
					__launchParseRequest(text);
				}
			}, 500);
		}
	};
	
	$input.addEventListener('keyup', (evt) => {
		if (evt.charCode === 13) {
			// 13 is the Enter key
			evt.preventDefault();
			$form.submit();
		} else {
			__checkInstantSearch();
		}
	});
	
	$form.addEventListener('submit', (evt) => {
		evt.preventDefault();
		const text = ($input.value.length > 0)
			? $input.value
			: DEFAULT_NLP_TEXT();
		updateURL(text);
		__launchParseRequest(text);
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


