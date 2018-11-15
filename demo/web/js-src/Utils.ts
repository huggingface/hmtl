
class Utils {
	static flatten<T>(arr: T[][]): T[] {
		return Array.prototype.concat.apply([], arr);
	}
}
