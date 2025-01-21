import nextra from 'nextra'

const withNextra = nextra({
  latex: true,
  search: {
    codeblocks: false
  },
  contentDirBasePath: '/docs'
})

export default withNextra({
  output: 'export',
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: '/',
        destination: '/docs',
        permanent: true,
      }
    ];
  },
})
