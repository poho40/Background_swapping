import React from 'react'

type headerProps = {
    
};

const header:React.FC<headerProps> = () => {
    
    return(
        <header className="shadow">
      <div className="relative z-20 bg-white">
        <div className="px-6 md:px-12 lg:container lg:mx-auto lg:px-6 lg:py-4">
          <div className="flex items-center justify-between">
            <div className="relative z-20 pb-1">
              {/* <Link href={ROUTES.HOME}><Image src="/images/logo.png" width={214} height={66} alt='Brix N Stones'/></Link> */}
            </div>

            <div className="flex items-center justify-end">
              <input
                type="checkbox"
                name="hamburger"
                id="hamburger"
                className="peer"
                hidden
              />
              <label
                htmlFor="hamburger"
                className="peer-checked:hamburger block relative z-20 p-6 -mr-6 cursor-pointer lg:hidden"
              >
                <div
                  aria-hidden="true"
                  className="m-auto h-0.5 w-6 rounded bg-sky-900 transition duration-300"
                ></div>
                <div
                  aria-hidden="true"
                  className="m-auto mt-2 h-0.5 w-6 rounded bg-sky-900 transition duration-300"
                ></div>
              </label>

              <div className="peer-checked:translate-x-0 fixed inset-0 w-[calc(100%-4.5rem)] translate-x-[-100%] bg-white shadow-xl transition duration-300 lg:border-r-0 lg:w-auto lg:static lg:shadow-none lg:translate-x-0">
                <div className="flex flex-col h-full justify-between lg:items-center lg:flex-row">
                  <ul className="px-6 pt-32 text-gray-700 space-y-8 md:px-12 lg:space-y-0 lg:flex lg:space-x-12 lg:pt-0">
                    <li className="pb-2 group relative before:absolute before:inset-x-0 before:bottom-0 before:h-2 before:origin-right before:scale-x-0 before:bg-zinc-300 before:transition before:duration-200 hover:before:origin-left hover:before:scale-x-100">
                      {/* <Link href={ROUTES.HOME}>Home</Link> */}
                    </li>
                    <li className="pb-2 group relative before:absolute before:inset-x-0 before:bottom-0 before:h-2 before:origin-right before:scale-x-0 before:bg-zinc-300 before:transition before:duration-200 hover:before:origin-left hover:before:scale-x-100">
                      {/* <Link href={ROUTES.ABOUT}>Studio & Team</Link> */}
                    </li>
                    <li className="pb-2 group relative before:absolute before:inset-x-0 before:bottom-0 before:h-2 before:origin-right before:scale-x-0 before:bg-zinc-300 before:transition before:duration-200 hover:before:origin-left hover:before:scale-x-100">
                      {/* <Link href={ROUTES.PROJECTS}>Our Journey</Link> */}
                    </li>
                    <li className="pb-2 group relative before:absolute before:inset-x-0 before:bottom-0 before:h-2 before:origin-right before:scale-x-0 before:bg-zinc-300 before:transition before:duration-200 hover:before:origin-left hover:before:scale-x-100">
                      {/* <Link href={ROUTES.CONTACT}>Get in Touch</Link> */}
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
    )
}
export default header;