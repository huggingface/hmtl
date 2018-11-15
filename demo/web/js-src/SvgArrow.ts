
interface ElOpts {
	classnames: string[];
	attributes: [string, string][];
	style:      [string, string][];
	children:   Element[];
	text:       string;
	id:         string;
	xlink:      string;
}

namespace SvgArrow {
	export interface Opts {
		label?:  string;
		marker?: string;
		/**
		 * "Height" of arrow.
		 * 0 means flat.
		 * 1 is low (the default)
		 * 3 is the max.
		 */
		height?: number;
	}
}

class SvgArrow {
	container:  HTMLElement;
	markFrom:   HTMLElement;
	markTo:     HTMLElement;
	label:      string;
	marker?:    string;
	height:     number;
	classNames: string[] = [];
	constructor(container: HTMLElement, markFrom: HTMLElement, markTo: HTMLElement, opts: SvgArrow.Opts = {}) {
		this.container = container;
		this.markFrom  = markFrom;
		this.markTo    = markTo;
		this.label     = opts.label || "";
		this.marker    = opts.marker;
		this.height    = opts.height || (this.label.length === 0) ? 3 : 1;
	}
	
	/// From displacy.js
	_el(tag: string, options: Partial<ElOpts> = {}): SVGElement {
		const { classnames = [], attributes = [], style = [], children = [], text, id, xlink } = options;
		const ns = 'http://www.w3.org/2000/svg';
		const nsx = 'http://www.w3.org/1999/xlink';
		const el = document.createElementNS(ns, tag);
		
		classnames.forEach(name => el.classList.add(name));
		attributes.forEach(([attr, value]) => el.setAttribute(attr, value));
		style.forEach(([ prop, value ]) => el.style[prop] = value);
		if (xlink) {
			el.setAttributeNS(nsx, 'xlink:href', xlink);
		}
		if (text) {
			el.appendChild(document.createTextNode(text));
		}
		if (id) {
			el.id = id;
		}
		children.forEach(child => el.appendChild(child));
		return el;
	}
	
	
	generate(): SVGElement {
		const rand = Math.random().toString(36).substr(2, 8);
		
		const startX = this.markFrom.getBoundingClientRect().left 
			- this.container.getBoundingClientRect().left
			+ this.markFrom.getBoundingClientRect().width / 2;
		
		const endX = this.markTo.getBoundingClientRect().left 
			- this.container.getBoundingClientRect().left
			+ this.markTo.getBoundingClientRect().width / 2;
		
		// const curveY = Math.max(-50, SvgArrow.yArrows - (endX - startX) / 3.2);
		const startY = this.container.querySelector('.text')!.getBoundingClientRect().top 
			- this.container.getBoundingClientRect().top
			- 2;
		
		const __heights = {
			0: 0,
			1: 10,
			2: 17,
			3: 24,
		};
		const height = __heights[this.height] || 10;
		const curveY = startY - height;
		
		const curve = (startX < endX)
			? `M${startX},${startY} C${startX},${curveY} ${endX},${curveY} ${endX},${startY}`
			: `M${endX},${startY} C${endX},${curveY} ${startX},${curveY} ${startX},${startY}`
		;
		
		const elPath = this._el('path', {
			id: `arrow-${rand}`,
			classnames: [ 'displacy-arc' ],
			attributes: [
				[ 'd', curve ],
				[ 'stroke-width', '1px' ],
				[ 'fill', 'none' ],
				[ 'stroke', 'currentColor' ],
			]
		});
		if (this.marker) {
			if (startX < endX) {
				elPath.setAttribute('marker-end',   `url(#marker-${this.marker})`);
			} else {
				elPath.setAttribute('marker-start', `url(#marker-${this.marker})`);
			}
		}
		
		return this._el('g', {
			classnames: [ 'displacy-arrow' ].concat(this.classNames),
			children: [
				elPath,
				this._el('text', {
					attributes: [
						[ 'dy', '-0.3em' ]
					],
					children: [
						this._el('textPath', {
							xlink: `#arrow-${rand}`,
							classnames: [ 'displacy-label' ],
							attributes: [
								[ 'startOffset', '50%' ],
								[ 'fill', 'currentColor' ],
								[ 'text-anchor', 'middle' ],
							],
							text: this.label
						})
					]
				}),
			]
		});
	}
	
	/**
	 * Generate the <defs> block for the arrowheads.
	 */
	static markersDefs(): SVGDefsElement {
		const el = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
		const colors = new Map<string, string>([
			[ 'relex-art',        '#4286f4' ],
			[ 'relex-gen-aff',    '#4df441' ],
			[ 'relex-org-aff',    '#cc2486' ],
			[ 'relex-part-whole', '#222526' ],
			[ 'relex-per-soc',    '#9924cc' ],
			[ 'relex-phys',       '#1ca02c' ],
		]);
		el.innerHTML = Array.from(colors.entries()).map(([k, v]) => {
			return `<marker id="marker-${k}" viewBox="0 0 10 10" refX="6" refY="6"
				markerWidth="6" markerHeight="6"
				orient="20">
				<path d="M 0 0 L 10 5 L 0 10 z" fill="${v}"/>
			</marker>`;
		}).join('');
		return el;
	}
}
