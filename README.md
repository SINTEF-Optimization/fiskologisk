## TODO:

### Client:
* Add display of metadata (max_total_biomass, "max_yearly_production")
* Set up proper deployment. The script run by gitlab should:
    * Run the "map_problems_for_frontend.sh" script
    * Run 'npm build'
    * Copy the contents of the client/fiskui/build folder to the public folder
    * Upload the public folder as an artifact