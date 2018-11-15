module.exports = function(grunt) {
	
	grunt.loadNpmTasks('grunt-contrib-less');
	grunt.loadNpmTasks('grunt-contrib-watch');
	grunt.loadNpmTasks('grunt-contrib-connect');
	
	grunt.registerTask('default', ['less']);
	
	grunt.initConfig({
		less: {
			options: {
				compress: true
			},
			dist: {
				src: "less/*.less",
				dest: "dist/style.css"
			}
		},
		watch: {
			options: {
				livereload: true
			},
			files: ["index.html", "less/*", "dist/*.js"],
			tasks: 'default'
		},
		connect: {
			server: {
				options: {
					keepalive: true,
					open: true,
					debug: true,
				}
			}
		}
	});
};
