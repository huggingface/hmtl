type EntityType = 
| 'GPE'
| 'ORG'
| 'PERSON'
| 'DATE'
| 'NORP'
| 'CARDINAL'
| 'MONEY'
| 'PERCENT'
| 'WORK_OF_ART'
| 'ORDINAL'
| 'EVENT'
| 'LOC'
| 'TIME'
| 'FAC'
| 'QUANTITY'
| 'LAW'
| 'PRODUCT'
| 'LANGUAGE'
;
type MentionType = 
| 'PER'
| 'GPE'
| 'ORG'
| 'FAC'
| 'LOC'
| 'WEA'
| 'VEH'
;
type RelationType =
| 'ORG-AFF'
| 'PHYS'
| 'ART'
| 'PER-SOC'
| 'PART-WHOLE'
| 'GEN-AFF'
;

interface Span {
	text:        string;
	begin_token: number;
	end_token:   number;
	begin_char:  number;
	end_char:    number;
}
interface Entity extends Span {
	type:        EntityType;
}
interface Mention extends Span {
	type:        MentionType;
}
interface Relation {
	type:            RelationType;
	arg1_text:       string;
	arg1_begin_char: number;
	arg1_end_char:   number;
	arg2_text:       string;
	arg2_begin_char: number;
	arg2_end_char:   number;
}
interface RelationHead extends Relation {
	arg1_index:      number;
	arg2_index:      number;
}
interface RelationExpanded extends Relation {
	arg1_begin_token: number;
	arg1_end_token:   number;
	arg2_begin_token: number;
	arg2_end_token:   number;
}
interface CorefArc {
	mention1_begin:      number;
	mention1_end:        number;
	mention1_begin_char: number;
	mention1_end_char:   number;
	text1:               string;
	mention2_begin:      number;
	mention2_end:        number;
	mention2_begin_char: number;
	mention2_end_char:   number;
	text2:               string;
}

interface NlpResponse {
	text:                   string;
	tokenized_text:         string;
	ner:                    Entity[];
	emd:                    Mention[];
	emd_expanded:           Mention[];
	relation_arcs:          RelationHead[];
	relation_arcs_expanded: RelationExpanded[];
	coref_arcs:             CorefArc[];
	coref_clusters:         Span[][];
}

type Task = 'ner' | 'emd' | 'relex' | 'coref';
const ALL_TASKS: Task[] = ['ner', 'emd', 'relex', 'coref'];


class HuggingNlp {
	endpoint: string;
	onStart   = () => {};
	onSuccess = () => {};
	
	constructor(endpoint: string, opts: any) {
		this.endpoint = endpoint;
		if (opts.onStart) {
			(<any>this).onStart   = opts.onStart;
		}
		if (opts.onSuccess) {
			(<any>this).onSuccess = opts.onSuccess;
		}
		window.addEventListener('resize', () => {
			this.svgResizeAll();
		});
	}
	
	container(task: Task): HTMLElement {
		return document.querySelector<HTMLElement>(`.task.${task} .container`)!;
	}
	svgContainer(task: Task): SVGSVGElement {
		return document.querySelector<SVGSVGElement>(`.task.${task} .svg-container`)!;
	}
	svgResizeAll() {
		for (const task of ALL_TASKS) {
			const container    = this.container(task);
			const svgContainer = this.svgContainer(task);
			svgContainer.setAttribute('width',  `${container.scrollWidth}`);   /// Caution: not offsetWidth.
			svgContainer.setAttribute('height', `${container.scrollHeight}`);
		}
	}
	
	parse(text: string) {
		this.onStart();
		
		const path = `${this.endpoint}?text=${encodeURIComponent(text)}`;
		const request = new XMLHttpRequest();
		request.open('GET', path);
		request.onload = () => {
			if (request.status >= 200 && request.status < 400) {
				this.onSuccess();
				const res: NlpResponse = JSON.parse(request.responseText);
				this.render(res);
			}
			else {
				console.error('Error', request);
			}
		};
		request.send();
	}
	dummyParse() {
		this.onStart();
		this.onSuccess();
		/// define `dummyJson` somewhere.
		this.render((<any>window).dummyJson);
	}
	
	
	render(res: NlpResponse) {
		const markupNER = Displacy.render(res.text, res.ner);
		this.container('ner').innerHTML = `<div class="text">${markupNER}</div>`;
		
		const markupEMD = Displacy.render(res.text, res.emd_expanded);
		this.container('emd').innerHTML = `<div class="text">${markupEMD}</div>`;
		
		const spansRELEX = Utils.flatten(res.relation_arcs_expanded.map(r => {
			return [
				{ type: 'SUBJ', begin_char: r.arg1_begin_char, end_char: r.arg1_end_char, index: `${r.arg1_begin_char}-${r.arg1_end_char}` },
				{ type: 'OBJ',  begin_char: r.arg2_begin_char, end_char: r.arg2_end_char, index: `${r.arg2_begin_char}-${r.arg2_end_char}` },
			]
		}));
		const markupRELEX = Displacy.render(res.text, spansRELEX);
		this.container('relex').innerHTML = `<div class="text">${markupRELEX}</div>`;
		
		const spansCOREF = Utils.flatten(res.coref_clusters.map((spans, i) => {
			return spans.map(s => {
				return Object.assign(
					{ type: `cluster-${i+1}`, index: `${s.begin_char}-${s.end_char}` },
					s
				);
			})
		}));
		const markupCOREF = Displacy.render(res.text, spansCOREF, ['dot']);
		this.container('coref').innerHTML = `<div class="text">${markupCOREF}</div>`;
		
		/// SVG
		this.svgContainer('relex').textContent = "";  // Empty
		this.svgContainer('coref').textContent = "";  // Empty
		this.svgResizeAll();
		
		/**
		 * Render arrows
		 */
		this.svgContainer('relex').appendChild(SvgArrow.markersDefs());
		for (const r of res.relation_arcs_expanded) {
			const markFrom = this.container('relex').querySelector<HTMLElement>(`mark[data-index="${r.arg1_begin_char}-${r.arg1_end_char}"]`)!;
			const markTo   = this.container('relex').querySelector<HTMLElement>(`mark[data-index="${r.arg2_begin_char}-${r.arg2_end_char}"]`)!;
			const arrow = new SvgArrow(
				this.container('relex'),
				markFrom,
				markTo,
				{
					label:  r.type,
					marker: `relex-${r.type.toLowerCase()}`,
				}
			);
			arrow.classNames = [ `relex-${r.type.toLowerCase()}` ];
			this.svgContainer('relex').appendChild(arrow.generate());
		}
		
		
		for (const r of res.coref_arcs) {
			const markFrom = this.container('coref').querySelector<HTMLElement>(`mark[data-index="${r.mention1_begin_char}-${r.mention1_end_char}"]`)!;
			const markTo   = this.container('coref').querySelector<HTMLElement>(`mark[data-index="${r.mention2_begin_char}-${r.mention2_end_char}"]`)!;
			/// Find out which cluster this arc belongs to.
			const clusterId = markFrom.dataset.entity || "";
			const arrow = new SvgArrow(
				this.container('coref'),
				markFrom,
				markTo
			);
			arrow.classNames = [ clusterId ];
			this.svgContainer('coref').appendChild(arrow.generate());
		}
	}
}
