

/**
 * Indicates position of spans of text inside the string. 
 * (for visual applications only, no semantic sense here.)
 */
/**
 * Modified version of coref-viz to account for different key names.
 */
interface DSpan {
	type:       string;
	begin_char: number;
	end_char:   number;
	index?:     string;
}

interface DSpanTag {
	span: DSpan;
	tag: "start" | "end";
}

class Displacy {
	static sortSpans(spans: DSpan[]) {
		spans.sort((a, b) => {  /// `a` should come first when the result is < 0
			if (a.begin_char === b.begin_char) {
				return b.end_char - a.end_char;   /// CAUTION.
			}
			return a.begin_char - b.begin_char;
		});
		
		// Check existence of **strict overlapping**
		spans.forEach((s, i) => {
			if (i < spans.length - 1) {
				const sNext = spans[i+1];
				if (s.begin_char < sNext.begin_char && s.end_char > sNext.begin_char) {
					console.log("ERROR", "Spans: strict overlapping");
				}
			}
		});
	}
	
	/**
	 * Render a text string and its entity spans
	 * 
	 * *see displacy-ent.js*
	 * see https://github.com/explosion/displacy-ent/issues/2
	 */
	static render(text: string, spans: DSpan[], classes?: string[]): string {
		this.sortSpans(spans);
		
		const tags: { [index: number]: DSpanTag[] } = {};
		const __addTag = (i: number, s: DSpan, tag: "start" | "end") => {
			if (Array.isArray(tags[i])) {
				tags[i].push({ span: s, tag: tag });
			} else {
				tags[i] = [{ span: s, tag: tag }];
			}
		};
		for (const s of spans) {
			__addTag(s.begin_char, s, "start");
			__addTag(s.end_char,   s, "end");
		}
		// console.log(JSON.stringify(tags));  // todo remove
		
		let out = {
			__content: "",
			append(s: string) {
				this.__content += s;
			}
		};
		let offset = 0;
		
		const indexes = Object.keys(tags).map(k => parseInt(k, 10)).sort((a, b) => a - b); /// CAUTION
		for (const i of indexes) {
			const spanTags = tags[i];
			// console.log(i, spanTags);  // todo remove
			if (i > offset) {
				out.append(text.slice(offset, i));
			}
			
			offset = i;
			
			for (const sT of spanTags) {
				if (sT.tag === "start") {
					out.append(
						(classes)
						? `<mark data-entity="${ sT.span.type.toLowerCase() }" data-index="${ sT.span.index || "" }" class="${ classes.join(' ') }">`
						: `<mark data-entity="${ sT.span.type.toLowerCase() }" data-index="${ sT.span.index || "" }">`
					);
					const singleScore: number | undefined = (<any>sT.span).singleScore;
					if (singleScore) {
						out.append(`<span class="single-score">${ singleScore.toFixed(3) }</span>`);
					}
				} else {
					out.append(`</mark>`);
				}
			}
		}
		
		out.append(text.slice(offset, text.length));
		return out.__content;
	}
}
